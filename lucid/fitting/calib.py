"""Calibration model and problem: the predicted per-sensor charge for every source × wavelength,
and the published estimator on it.

:class:`CalibrationForward` is the model, the calibration counterpart of
:class:`lucid.fitting.recon.ReconModel`. :class:`CalibrationProblem` adds the observed charge and
the estimator (Neyman residual, profiled gains, the key streams) and is what a solver runs on.

Preserved expression for expression from the paper's reference calibration engine, its keys
aside. The whole stack runs in float32 (``jax_enable_x64`` is never enabled), so an equivalent but
re-associated expression can move the result. The reference engine is not in this repository, so
no shipped test gates that agreement. Anything here that looks gratuitously specific (the
concatenate-then-transpose, the laser/isotropic split) is load-bearing for that reason.

Keys
----
Every random stream starts from ``fold_in(PRNGKey(seed), role)``, one role per stream
(:data:`ROLE_TRUTH` for :func:`~lucid.fitting.calibrate.closure_data`, :data:`ROLE_FORWARD` and
:data:`ROLE_JACOBIAN` for :class:`CalibrationProblem`, :data:`ROLE_READOUT` for a draw outside
the fit), and folds in its indices one at a time: step, then draw, then source, then wavelength.
No index is added to another, so no two (seed, role, step, draw, source, wavelength) share a key.

Layout
------
``S = W * n_sources`` configurations, indexed ``g = wl * n_sources + s`` — wavelength-major, so
each row of the returned ``(S, NS)`` array is one (wavelength, source) pair.

Dispatch
--------
Each source reaches the simulator as a **pytree of arrays**, so one compiled program serves every
source of one pytree structure with the source as a traced argument — that is what makes a
``pmap`` over sources possible at all.

Sharding
--------
The reference dispatches source 0 (a collimated laser) to one device and the remaining isotropic
sources to a ``pmap``. That is an execution strategy, not physics, so it is injected via
``map_fn`` rather than baked in. The default is a serial loop over sources, which is also what
the reference itself falls back to below 7 devices.
"""

import functools

import jax
import jax.numpy as jnp

__all__ = [
    "CalibrationForward",
    "CalibrationProblem",
    "PulseResponse",
    "DigitResponse",
    "profile_gains",
    "jackknife_profile",
    "neyman_residual",
]

ROLE_TRUTH, ROLE_FORWARD, ROLE_JACOBIAN, ROLE_READOUT = range(4)


class CalibrationForward:
    """Mean per-sensor charge at fit vector ``theta``, for every source × wavelength.

    Parameters
    ----------
    sim : callable
        ``sim(source, detector_params, key) -> (charge, time)``, from
        ``setup_event_simulator(..., source='light_source')`` (first-arrival hits, expected).
    sources : sequence
        Calibration sources. Source 0 is treated as the "singleton" (the laser in the reference
        layout) and the rest as the mappable group — this only affects dispatch, not the result.
    params : parameterisation
        Supplies ``W`` and ``to_dp`` (:mod:`lucid.fitting.params`).
    n_sensors : int or None
        Defaults to the simulator's, ``sim.det_geom.num_sensors``.
    map_fn : callable or None
        ``map_fn(fn, in_axes)`` returning a batched callable over the non-singleton sources.
        ``None`` (default) uses a serial loop that is numerically identical.
    predict : callable or None
        ``predict(theta, source, wl, gains, key) -> (n_sensors, ...)``, replacing the default
        composition ``sim(source, params.to_dp(theta, wl, gains), key)[0]``. Giving it selects the
        per-source route: sources are then opaque Python objects rather than traced pytrees, so
        each gets its own compiled program and ``map_fn`` does not apply. This is how a list of
        ready-made forward closures is fitted. It may return trailing axes when a ``response``
        turns them into the charge (:class:`DigitResponse`: the expected windows per number of
        photoelectrons); every array then carries them, ``(S, NS, ...)``. ``n_sensors`` is then
        required, and ``params`` may be None (one wavelength).
    response : PulseResponse, DigitResponse or None
        The digitizer in the model: the charge compared with the data is ``response``'s recorded
        charge at the profiled gains rather than the gain times the forward (see
        :class:`CalibrationProblem`). None: the gain times the forward.
    source_columns : sequence of index arrays or None
        Per-source route only: for each source, the entries of ``theta`` that can move its charge
        (the shared parameters and its own). Its Jacobian is then taken with respect to those
        alone, which cuts the tangent batch and its memory; the other columns are exactly zero.
    """

    def __init__(
        self,
        sim,
        sources,
        params,
        n_sensors=None,
        *,
        map_fn=None,
        laser_device=None,
        predict=None,
        response=None,
        source_columns=None,
    ):
        self.sim = sim
        self.sources = list(sources)
        self.params = params
        if n_sensors is None and sim is None:
            raise ValueError("n_sensors is required without a simulator")
        self.NS = int(sim.det_geom.num_sensors if n_sensors is None else n_sensors)
        self.W = 1 if params is None else params.W
        self.n_sources = len(self.sources)
        self.S = self.W * self.n_sources
        self.map_fn = map_fn
        self.response = response
        self._per_source = None
        charge_at = getattr(response, "predict_charge", None)
        if source_columns is not None:
            if predict is None:
                raise ValueError("source_columns needs the per-source route (predict=...)")
            if len(source_columns) != self.n_sources:
                raise ValueError(
                    f"source_columns has {len(source_columns)} entries for {self.n_sources} sources"
                )
        if charge_at is not None and predict is None:
            raise ValueError(
                "a response with predict_charge needs the per-source route (predict=...)"
            )

        def model(mu, lk, wl):
            # The model charge compared with the data: the gain times the forward, through the
            # response when there is one. Inside `sm`, so the response is differentiated in the
            # same fused expression as the weight.
            if response is None:
                return jnp.exp(lk) * mu
            return response.model_charge(mu, jnp.exp(lk), wl)

        if predict is not None:
            cols = [None] * self.n_sources if source_columns is None else source_columns
            self._per_source = [
                self._compile_one(predict, src, model, charge_at, c)
                for src, c in zip(self.sources, cols)
            ]
            return

        def _mbody(theta, src, keys, gains):  # (W, NS); vmap over wavelengths
            return jax.vmap(lambda wl, k: sim(src, params.to_dp(theta, wl, gains), k)[0])(
                jnp.arange(self.W), keys
            )

        def _jbody(theta, src, wl, lk, key, q_cfg, q_floor):  # (NS, P); see `jacobian`
            def sm(th):
                mu = sim(src, params.to_dp(th, wl, jnp.ones(self.NS)), key)[0]
                return model(mu, lk, wl) / jnp.sqrt(jnp.clip(q_cfg, q_floor, None))

            return jax.jacfwd(sm)(theta)

        self._mbody = _mbody
        self._single = jax.jit(_mbody)
        self._jsingle = jax.jit(_jbody)
        # Where the singleton source runs. `map_fn` pmaps the grouped sources over devices
        # 0..n_group-1, and an un-placed `_single` runs on JAX's default device (device 0), so
        # the laser would serialise behind pmap shard 0 instead of overlapping it. Placement is
        # value-neutral: it buys wall clock and moves no number. `laser_device` overrides; None
        # picks the first card outside the mapped range, falling back to the last available when
        # there is none.
        self._laser_dev = laser_device
        if self._laser_dev is None and self.n_sources > 1:
            devs = jax.devices()
            self._laser_dev = devs[min(self.n_sources - 1, len(devs) - 1)]
        # Sources are grouped by pytree structure and each group is mapped over its own stack;
        # singletons are called directly. This admits mixed layouts (e.g. a `LaserSource` beside
        # `IsotropicSource`s), which source diversity in calibration needs. The published layout
        # still resolves to one singleton (the laser) plus one mapped group of seven: the same two
        # calls, in the same order, on the same keys, so it is bit-exact by construction.
        self._groups = _group_by_structure(self.sources)
        # Parts get disjoint devices. A single-source part runs unstacked on a card outside the
        # mapped range, so it runs alongside the mapped parts rather than queueing behind one of
        # their shards. In the published layout that part is the laser, but nothing here knows
        # what a laser is.
        self._placed = next((ix[0] for ix, _ in self._groups if len(ix) == 1), None)
        self._largest = max((len(ix) for ix, _ in self._groups), default=0)
        # The Jacobian maps over the SAME groups: its rows are assembled in source order to match
        # the forward's, so a different partition would pair row `g` with another source's column.
        self._group_fns, self._jgroup_fns, self._group_stacks = [], [], []
        for idx, stack in self._groups:
            if len(idx) == 1:
                self._group_fns.append(None)  # singleton: use self._single
                self._jgroup_fns.append(None)
                self._group_stacks.append(None)
                continue
            self._group_fns.append(self._make_map(_mbody, (None, 0, 0, None), len(idx)))
            self._jgroup_fns.append(
                self._make_map(_jbody, (None, 0, None, None, 0, 0, None), len(idx))
            )
            self._group_stacks.append(stack)

    def _make_map(self, fn, in_axes, n):
        """A mapped callable over `n` items, honouring a caller-supplied `map_fn`.

        `map_fn(fn, in_axes)` is the established two-argument contract. A `pmap` built from it
        maps over whatever leading axis it is handed, so it serves any group; only the serial
        fallback needs the count `n`.
        """
        if self.map_fn is None:
            return _serial_map(fn, in_axes, n)
        return self.map_fn(fn, in_axes)

    def _compile_one(self, predict, src, model, charge_at, cols):
        """The per-source route's two compiled programs for ``src``: the forward
        ``(theta, keys, gains) -> (W, NS, ...)`` and the Jacobian body (see :meth:`jacobian`)."""
        W, NS = self.W, self.NS

        def body(theta, keys, gains):
            return jax.vmap(lambda wl, k: predict(theta, src, wl, gains, k))(jnp.arange(W), keys)

        def jbody(theta, wl, lk, key, q_cfg, q_floor):
            def sm(th):
                if charge_at is not None:
                    # The response's own charge at the gains, whose derivative is cheaper than
                    # the windows' (DigitResponse.predict_charge).
                    q = charge_at(th, src, wl, jnp.exp(lk), key)
                else:
                    q = model(predict(th, src, wl, jnp.ones(NS), key), lk, wl)
                return q / jnp.sqrt(jnp.clip(q_cfg, q_floor, None))

            if cols is None:
                return jax.jacfwd(sm)(theta)
            # Only the entries `cols` of theta can move this source's charge: differentiate with
            # respect to them alone and scatter into the full width (every other column is zero).
            c = jnp.asarray(cols, dtype=jnp.int32)
            sub = jax.jacfwd(lambda s: sm(theta.at[c].set(s)))(theta[c])
            return jnp.zeros(sub.shape[:-1] + theta.shape, sub.dtype).at[..., c].set(sub)

        return jax.jit(body), jax.jit(jbody, static_argnums=(1,))

    def keys(self, key):
        """The ``(n_sources, W)`` keys of one draw on ``key``: ``fold_in(fold_in(key, s), wl)``.

        Indexed by the source's ABSOLUTE position, so a source's random stream does not depend on
        how the layout happens to be grouped.
        """
        return _fold_grid(key, self.n_sources, self.W)

    def __call__(self, theta, key, gains):
        """-> ``(S, NS)`` mean charge, row ``g = wl * n_sources + s``, on the draw ``key``
        (``(S, NS, ...)`` when ``predict`` returns trailing axes)."""
        allk = self.keys(key)
        if self._per_source is not None:
            per = jnp.stack(
                [fwd(theta, allk[i], gains) for i, (fwd, _) in enumerate(self._per_source)]
            )  # (n, W, NS, ...)
            # Swapping the first two axes is the (1, 0, 2) transpose below, any trailing axes kept.
            return jnp.swapaxes(per, 0, 1).reshape((self.S, self.NS) + per.shape[3:])

        slots = [None] * self.n_sources
        for (idx, _), fn, stack in zip(self._groups, self._group_fns, self._group_stacks):
            if fn is None:  # singleton
                i = idx[0]
                # Put a singleton on its own card so it overlaps a grouped pmap rather than
                # queueing behind shard 0 (see __init__).
                d = self._laser_dev if i == self._placed else None
                if d is None:
                    slots[i] = self._single(theta, self.sources[i], allk[i], gains)
                else:
                    slots[i] = self._single(
                        jax.device_put(theta, d),
                        jax.device_put(self.sources[i], d),
                        jax.device_put(allk[i], d),
                        jax.device_put(gains, d),
                    )
                continue
            out = fn(theta, stack, allk[jnp.asarray(idx)], gains)  # (len(idx), W, NS)
            for j, i in enumerate(idx):
                slots[i] = out[j]
        per = jnp.stack([jnp.asarray(v) for v in slots])  # (n, W, NS), SOURCE order
        return jnp.transpose(per, (1, 0, 2)).reshape(self.S, self.NS)

    def average(self, theta, key, gains, n_draws):
        """Mean over ``n_draws`` independent forward draws, draw ``b`` on ``fold_in(key, b)``.

        Averaging reduces the forward's Monte-Carlo variance. Were the residual linear in ``M``,
        ``E[r(M)] = r(E[M])`` and the fixed point would sit at truth at any variance. What makes
        ``n_draws`` matter is the profiled gain ``k = ΣQ/ΣM``: it is nonlinear in ``M``, so its
        expectation shifts with the forward's variance, and that shift propagates into the
        residual. The bias and its scaling with forward noise are measured in
        ``tests/fitting/test_fitting_estimator_unbiased.py``.
        """
        acc = None
        for b in range(n_draws):
            d = self(theta, jax.random.fold_in(key, b), gains)
            # zeros + d: the sum a zeros start always gave, for any trailing axes.
            acc = jnp.zeros(d.shape) + d if acc is None else acc + d
        return acc / n_draws

    def draws(self, theta, key, gains, n_draws):
        """The ``n_draws`` draws :meth:`average` averages, each kept: ``(n_draws, S, NS, ...)``.

        Same keys as :meth:`average`, for :func:`jackknife_profile`, which needs them apart.
        """
        return jnp.stack([self(theta, jax.random.fold_in(key, b), gains) for b in range(n_draws)])

    def jacobian(self, theta, lk, key, data, q_floor, draws=(0,)):
        """``∂r/∂θ`` of the Neyman residual, ``(S, NS, P)``, rows ordered like the forward's.

        Differentiated as ONE fused expression. It does not compute ``∂μ/∂θ`` and then apply the
        ``1/√Q`` weight; it runs ``jacfwd`` over the *already weighted* model::

            sm(θ) = exp(lk) · sim(src, dp(θ), key) / √clip(Q, floor)
            J     = jacfwd(sm)(θ)

        Splitting that into a model Jacobian followed by a scaling is algebraically the same and
        **numerically different in float32**, which is the only precision this stack ever runs
        in. So the weight stays inside the differentiated function, and the data enter the
        Jacobian rather than only the residual.

        Two further properties are deliberate, not incidental:

        * the gains are passed as ones and the per-PMT factor enters as the constant ``exp(lk)``
          multiplier, so ``∂k/∂θ`` is dropped by construction: the gains are profiled, not
          fitted;
        * :class:`CalibrationProblem` computes ``lk`` from the *residual's* draw, so J and r
          share that one scalar per sensor even though their photon streams are disjoint. It
          vanishes exactly when the gains are held fixed. The magnitude of the residual coupling
          has not been measured; do not assume it small.

        Draw ``h`` runs on ``keys(fold_in(key, h))``, as :meth:`average`'s draws do. ``draws`` is
        the set of draw indices to average over, not merely a count, so that disjoint halves can
        be requested independently.
        """
        draws = tuple(draws)
        nh = len(draws)
        draw_keys = {h: self.keys(jax.random.fold_in(key, h)) for h in draws}
        if self._per_source is not None:
            rows = []
            for wl in range(self.W):
                base = wl * self.n_sources
                rows.append(
                    jnp.stack(
                        [
                            sum(
                                jac(theta, wl, lk, draw_keys[h][i, wl], data[base + i], q_floor)
                                for h in draws
                            )
                            / nh
                            for i, (_, jac) in enumerate(self._per_source)
                        ]
                    )
                )
            return jnp.concatenate(rows, axis=0)
        rows = []
        for wl in range(self.W):
            base = wl * self.n_sources
            slots = [None] * self.n_sources
            for (idx, _), fn, stack in zip(self._groups, self._jgroup_fns, self._group_stacks):
                if fn is None:  # singleton
                    i = idx[0]
                    # Placed like the forward's singleton (see __init__). Value-neutral.
                    d = self._laser_dev if i == self._placed else None
                    if d is None:
                        slots[i] = (
                            sum(
                                self._jsingle(
                                    theta,
                                    self.sources[i],
                                    wl,
                                    lk,
                                    draw_keys[h][i, wl],
                                    data[base + i],
                                    q_floor,
                                )
                                for h in draws
                            )
                            / nh
                        )
                    else:
                        th_l, lk_l = jax.device_put(theta, d), jax.device_put(lk, d)
                        src_l = jax.device_put(self.sources[i], d)
                        q_l = jax.device_put(data[base + i], d)
                        slots[i] = (
                            sum(
                                self._jsingle(
                                    th_l,
                                    src_l,
                                    wl,
                                    lk_l,
                                    jax.device_put(draw_keys[h][i, wl], d),
                                    q_l,
                                    q_floor,
                                )
                                for h in draws
                            )
                            / nh
                        )
                    continue
                out = (
                    sum(
                        fn(
                            theta,
                            stack,
                            wl,
                            lk,
                            draw_keys[h][jnp.asarray(idx), wl],
                            jnp.stack([data[base + i] for i in idx]),
                            q_floor,
                        )
                        for h in draws
                    )
                    / nh
                )
                for j, i in enumerate(idx):
                    slots[i] = out[j]
            rows.append(jnp.stack([jnp.asarray(v) for v in slots]))  # (n, NS, P)
        return jnp.concatenate(rows, axis=0)


def profile_gains(model_charge, observed_sum, gauge="log", clip_min=1e-6):
    """Closed-form per-PMT gains ``k = ΣQ / ΣM``, gauged — the nuisance, solved not fitted.

    There is one gain per sensor (order 10^4 of them). Fitting them jointly would swamp the
    handful of optical parameters that are actually of interest, so they are profiled out at every
    step in closed form, at the cost of one division.

    ``ΣQ/ΣM`` is the exact minimiser of an UNWEIGHTED least squares (equivalently the Poisson
    scale MLE). It is NOT the minimiser of the Neyman chi-square this module actually forms in
    :func:`neyman_residual`, which is ``ΣM / Σ(M²/Q)``. Worked example: ``M=(1,10)``, ``Q=(2,10)``
    gives ``k=1.0909`` and ``chi2=0.4959`` here against ``k=1.0476`` and ``chi2=0.4762`` at the
    true Neyman minimum -- 4.1% higher. Two consequences follow and neither is cosmetic. The
    reported ``loss`` is not the profiled minimum; and since ``dchi2/dk = +0.909 != 0`` at this
    ``k``, the envelope theorem does NOT hold, so dropping ``dk/dtheta`` from the Jacobian (a
    deliberate choice, documented at :meth:`CalibrationForward.jacobian`) omits a genuinely non-zero
    term rather than a vanishing one.

    This is the published estimator and it is consistent -- ``E[r] = 0`` at truth for any gain
    map -- so the discrepancy is recorded rather than fixed. Changing ``k`` would move every
    published number.

    The gauge removes the exact degeneracy between a global gain and the overall light yield —
    without it the two directions are unidentifiable.

    ``'log'``    ``mean(log k) = 0``  — what the published run used.
    ``'linear'`` ``mean(k) = 1``      — ``k̂ = ΣQ/ΣM`` is linear in the data and therefore
                 unbiased, whereas taking its log first incurs a Jensen shift that is larger on
                 dim sensors. Not the published choice; adopting it would move published numbers.
    """
    return _gauge_gains(jnp.clip(observed_sum / model_charge, clip_min, None), gauge)


def _gauge_gains(k, gauge):
    """Apply the gain gauge (see :func:`profile_gains`); ``None`` leaves the gains ungauged."""
    if gauge is None:
        return k
    if gauge == "linear":
        return k / jnp.mean(k)
    if gauge == "log":
        return jnp.exp(jnp.log(k) - jnp.log(k).mean())
    raise ValueError(f"gauge must be 'log' or 'linear', got {gauge!r}")


def jackknife_profile(draws, observed_sum, gauge="log", response=None, wl=None, clip_min=1e-6):
    """Profiled gains and model charge from ``n >= 2`` forward draws, with their 1/n bias removed.

    Profiling the gains, ``k = ΣQ / ΣM``, puts the forward's Monte-Carlo charge in a denominator,
    so the model charge compared with the data, ``k·M``, is a ratio of noisy charges. Its expectation sits
    ``∝ var(M) ∝ 1/n`` off its value at the mean model (Jensen), most on dim sensors: the bias
    ``tests/fitting/test_fitting_estimator_unbiased.py`` measures. The jackknife cancels that term from the
    draws already in hand. For any estimate ``f`` made from all ``n`` draws and ``f_(-i)`` made
    from the ``n - 1`` others,

        f_JK = n·f − (n − 1)·mean_i f_(-i),

    leaving O(1/n²). It is applied to the model charge, which is what enters the (linear) residual;
    jackknifing the gains alone does not remove the bias, because the model carries the
    covariance of the gains with the charges they multiply. The gains are jackknifed too, for the
    gain map a calibration reports and the Jacobian.

    ``draws`` is ``(n, S, NS)`` (:meth:`CalibrationForward.draws`). With a ``response``
    (:class:`PulseResponse`, :class:`DigitResponse`) each gain is its Newton profile on the
    response's model, ``wl`` its rows' wavelength indices, and the model the response's charge
    (``draws`` then ``(n, S, NS, ...)`` with the response's trailing axes). Returns ``(k, model)``.
    """
    n = draws.shape[0]
    if n < 2:
        raise ValueError(f"the jackknife needs at least two forward draws, got {n}")
    total = draws.sum(0)
    loo = (total[None] - draws) / (n - 1)  # each draw left out

    def profiled(m):
        if response is None:
            k = profile_gains(m.sum(0) + 1e-12, observed_sum, gauge=gauge, clip_min=clip_min)
            return k, k[None, :] * m
        k = response.profile_gains(m, observed_sum, wl, gauge=gauge, clip_min=clip_min)
        return k, response.model_charge(m, k, wl)

    k_all, model_all = profiled(total / n)
    k_loo, model_loo = jax.vmap(profiled)(loo)
    k = jnp.clip(n * k_all - (n - 1) * k_loo.mean(0), clip_min, None)
    return k, n * model_all - (n - 1) * model_loo.mean(0)


def neyman_variance(data, q_floor, data_var=None):
    """The per-sensor variance in the Neyman weight: ``clip(Q, floor)``.

    ``data_var``, when given, replaces ``Q`` as the variance (still clipped at the floor): the
    measured variance of each sensor's charge, e.g. its pulse-to-pulse spread for a pulsed source,
    where digitized charge is not Poisson in photoelectrons. The weight still depends only on
    the data, not the model.
    """
    return jnp.clip(data if data_var is None else data_var, q_floor, None)


def neyman_residual(model_charge, data, q_floor, data_var=None):
    """``r = (k·M − Q) / √clip(Q, floor)`` — see :func:`neyman_variance`.

    The weight depends on the DATA only. That is the whole point: the forward model is a
    Monte-Carlo estimate redrawn every step, so a weight involving ``M`` would make the residual
    nonlinear in it and bias the gradient — permanently displacing the fixed point rather than
    merely adding noise. Neyman is the member of the χ² family whose weight is data-only.
    """
    return (model_charge - data) / jnp.sqrt(neyman_variance(data, q_floor, data_var))


class PulseResponse:
    """The digitizer and dark noise in the calibration model, for a pulsed source.

    A pulse gives a sensor ``x`` expected photoelectrons, for which the digitizer records a smooth
    mean charge ``g(x)``: the threshold curve, SPE spectrum and charge precision averaged over
    Poisson(``x``) photoelectrons (:func:`lucid.simulation.digitizer.pulse_charge_response`). Dark
    pulses add a pedestal. Over ``n_pulses`` pulses the model charge is
    ``n_pulses * (g(x) + pedestal)`` with ``x = pulse_scale[wl] * k * mu``: ``mu`` the forward's
    charge, ``k`` the sensor's gain -- QE-like, so inside the response, as :func:`closure_data`
    bakes it into the simulation -- and ``pulse_scale[wl]`` the expected photoelectrons per pulse
    per unit of forward charge at wavelength ``wl`` (photons per pulse / the sources' intensity).

    ``g`` is tabulated once, as the ratio ``g(x) / x`` on a log grid (between 0.83 and 1 for the
    HK digitizer), and interpolated in JAX: the model stays differentiable, and exactly linear
    below the grid. Above ``x_max`` the ratio is held. Not modelled: dead time and light later
    than the integration window (see the digitizer's pulsed-source functions).
    """

    def __init__(
        self, model, n_pulses, pulse_scale, dark_mu=0.0, x_max=300.0, n_grid=600, rng=None
    ):
        import numpy as np

        from lucid.simulation.digitizer import pulse_charge_response

        grid = np.geomspace(1e-6, float(x_max), int(n_grid))
        mean, _, _ = pulse_charge_response(
            model, np.concatenate([[0.0], grid]), dark_mu=dark_mu, rng=rng
        )
        self.model = model
        self.n_pulses = int(n_pulses)
        self.dark_mu = float(dark_mu)
        self.pedestal = float(mean[0])
        self.pulse_scale = jnp.asarray(pulse_scale, dtype=jnp.float32)
        self._x_min = float(grid[0])
        self._log_grid = jnp.asarray(np.log(grid), dtype=jnp.float32)
        self._ratio = jnp.asarray((mean[1:] - mean[0]) / grid, dtype=jnp.float32)

    def per_pulse(self, x):
        """``g(x)``: the mean recorded prompt charge of one pulse at ``x`` expected photoelectrons."""
        return x * jnp.interp(jnp.log(jnp.maximum(x, self._x_min)), self._log_grid, self._ratio)

    def charge(self, mu, wl):
        """Model charge over the run for forward charge ``mu`` (gain applied) at wavelength ``wl``."""
        return self.n_pulses * (self.per_pulse(self.pulse_scale[wl] * mu) + self.pedestal)

    def model_charge(self, mu, k, wl):
        """The response interface :class:`CalibrationProblem` uses: the charge at gains ``k``
        ``(NS,)`` for the forward's ``mu``. Here the gain multiplies the forward: ``charge(k * mu)``."""
        return self.charge(k * mu, wl)

    def profile_gains(self, mu, observed_sum, wl, gauge="log", clip_min=1e-6, n_newton=8):
        """Per-sensor gains that match the model's charge, summed over configurations, to the
        data's -- :func:`profile_gains`' ``k = ΣQ/ΣM`` for a model nonlinear in ``k``.

        ``g`` is monotone and nearly linear, so Newton's method from the linear solution converges
        in a few steps; ``n_newton`` are taken, vectorized over sensors. ``wl`` is each row's
        wavelength index, ``(S, 1)``. The gauge is then applied as in :func:`profile_gains`.
        """
        x = self.pulse_scale[wl] * mu  # (S, NS), per unit gain
        target = observed_sum / self.n_pulses - mu.shape[0] * self.pedestal
        k = jnp.clip(target / (x.sum(0) + 1e-12), clip_min, None)

        def step(k, _):
            f, df = jax.jvp(
                lambda kk: self.per_pulse(kk[None, :] * x).sum(0), (k,), (jnp.ones_like(k),)
            )
            return jnp.clip(k - (f - target) / (df + 1e-12), clip_min, None), None

        k, _ = jax.lax.scan(step, k, None, length=n_newton)
        return _gauge_gains(k, gauge)


class DigitResponse:
    """The digitiser's expected recorded charge in the calibration model, with the gain inside it.

    The forward (``predict``) returns, per sensor, the expected number of integration windows that
    hold ``m`` photoelectrons over the run, ``W[..., m-1]`` for ``m = 1 .. M``
    (:func:`lucid.simulation.hits.digit_windows`, times the number of pulses). They depend
    on the photoelectron rate alone: a sensor's gain scales its pulse heights, not its photoelectrons.
    The charge a sensor records is then

        Q = sum_m W_m c_m(g),

    ``c_m(g)`` the expected recorded charge of a window of ``m`` photoelectrons at gain ``g``: the
    discriminator and the charge step act on the scaled pulse height, so ``c_m(g)`` is not ``g c_m(1)``
    (for the HK digitiser a one-photoelectron window goes as about ``g^1.25``, a many-photoelectron one
    as about ``g``). A single factor per sensor, :func:`profile_gains`' ``k = ΣQ/ΣM`` on the gain-1
    charge, therefore cannot match a sensor whose window mix differs between configurations; this
    response profiles ``g`` itself.

    ``c_m(g)`` is tabulated on ``ln_g`` ``(G,)`` as ``table`` ``(G, M + 1)`` (column 0, the empty window,
    unused), linear in ``ln g`` between the points and held beyond them.

    ``predict_charge``, optional: ``(theta, source, wl, gains, key) -> (NS,)`` the charge ``Q`` at given
    gains, computed from the same forward, with a derivative cheaper than the windows' (the windows
    have one output per ``m``; their sum against ``c_m(g)`` has one). :meth:`CalibrationForward.jacobian`
    differentiates it, at the profiled gains, in place of ``predict`` followed by :meth:`model_charge`.
    """

    def __init__(self, ln_g, table, predict_charge=None):
        self._ln_g = jnp.asarray(ln_g, dtype=jnp.float32)
        self._table = jnp.asarray(table, dtype=jnp.float32)
        if self._table.ndim != 2 or self._table.shape[0] != self._ln_g.shape[0]:
            raise ValueError(f"table {self._table.shape} does not match ln_g {self._ln_g.shape}")
        self._d = self._ln_g[1] - self._ln_g[0]
        self.predict_charge = predict_charge

    def c_of(self, g):
        """``(..., M + 1)``: ``c_m`` at gains ``g``, linear in ``ln g`` between the table's points, held beyond."""
        pos = jnp.clip((jnp.log(g) - self._ln_g[0]) / self._d, 0.0, self._ln_g.shape[0] - 1 - 1e-6)
        i0 = pos.astype(jnp.int32)
        f = (pos - i0)[..., None]
        return self._table[i0] * (1.0 - f) + self._table[i0 + 1] * f

    def model_charge(self, W, k, wl):
        """The charge at gains ``k`` ``(NS,)`` for windows ``W`` ``(..., NS, M)``: ``sum_m W_m c_m(k)``."""
        M = W.shape[-1]
        return (W * self.c_of(k)[..., 1 : M + 1]).sum(-1)

    def profile_gains(self, W, observed_sum, wl, gauge="log", clip_min=1e-6, n_newton=10):
        """Per-sensor gains ``g`` with ``sum_configs sum_m W_m c_m(g)`` equal to the data's sum:
        :func:`profile_gains`' ``k = ΣQ/ΣM`` for a charge nonlinear in the gain.

        ``W`` is ``(S, NS, M)``. Newton's method in ``ln g`` from the linear solution (the gain-1 charge
        scaled to the data), vectorized over sensors; ``Q`` rises monotonically with ``g``, and the
        step is clipped to the table's range, where ``c_m`` is held and the slope would vanish. The
        gauge is then applied as in :func:`profile_gains`.
        """
        Ws = W.sum(0)  # (NS, M): over configurations
        lo, hi = self._ln_g[0], self._ln_g[-1]

        def q(lg):
            return self.model_charge(Ws, jnp.exp(lg), wl)

        lg = jnp.clip(
            jnp.log(jnp.clip(observed_sum / (q(jnp.zeros(Ws.shape[0])) + 1e-12), clip_min, None)),
            lo,
            hi,
        )

        def step(lg, _):
            f, df = jax.jvp(q, (lg,), (jnp.ones_like(lg),))
            return jnp.clip(lg - (f - observed_sum) / jnp.maximum(df, 1e-12), lo, hi), None

        lg, _ = jax.lax.scan(step, lg, None, length=n_newton)
        return _gauge_gains(jnp.clip(jnp.exp(lg), clip_min, None), gauge)


class CalibrationProblem:
    """A model and the observed charge: the published estimator, ready for any solver.

    The estimator is the Neyman residual ``(k·M − Q)/√clip(Q, floor)`` with the per-PMT gains
    ``k`` profiled in closed form (:func:`profile_gains`). Every method is a plain function of the
    fit vector ``theta`` and a ``step`` index, which selects the Monte-Carlo draw: the forward is
    redrawn at every step, so a fixed ``step`` gives a deterministic objective (common random
    numbers) and successive steps give independent draws.

    Two things this class owns that a solver must not:

    * **The keys.** The residual's forward draws from one stream (:data:`ROLE_FORWARD`) and the
      Jacobian from another (:data:`ROLE_JACOBIAN`), which must be independent of it (shared keys
      correlate J with r) and must carry the seed.
      Without a seed term in the Jacobian's stream every ensemble member draws the SAME Jacobian
      noise, so any fixed point it displaces moves them all alike: the error appears as bias and
      contributes nothing to the spread, leaving the ensemble's own error bar blind to it.
    * **The nuisance.** The gains are profiled each step, so they never enter ``theta``. This is
      why calibration needs no Schur block in the fitter, and why one loop serves calibration and
      reconstruction.

    Parameters
    ----------
    forward : CalibrationForward
    data : array ``(S, n_sensors)``
        Observed charge, rows ordered ``g = wl*n_sources + s`` to match the forward. With a single
        wavelength that is one row per source.
    seed : int
        Selects both key streams. Vary it to build an ensemble.
    q_floor_frac : float
        Floor on the Neyman weight's denominator, as a fraction of ``mean(data)``. It exists
        because ``1/sqrt(Q)`` diverges on an empty sensor: without it a handful of nearly unlit
        sensors dominate the fit.
    gauge : ``'log'`` or ``'linear'``
        Gain gauge; see :func:`profile_gains`.
    n_forward_draws : int
        Forward draws averaged into the residual each step. The Neyman residual is linear in
        ``M``, so on its own averaging would not move the fixed point; it matters because the
        profiled gain ``k = ΣQ/ΣM`` is nonlinear in ``M``
        (``tests/fitting/test_fitting_estimator_unbiased.py`` measures the bias).
    jacobian_draws : int
        Jacobian draws averaged per refresh, on the independent stream.
    data_var : array ``(S, n_sensors)`` or None
        The variance in the Neyman weight in place of ``data`` itself (see
        :func:`neyman_variance`): the measured variance of the data (a pulsed source's
        pulse-to-pulse spread, :func:`~lucid.fitting.calibrate.pulsed_closure_data`), or a
        variance held fixed over a fit (an IRLS stage's model charge). None: ``data``.

    With ``forward.response`` (:class:`PulseResponse`, :class:`DigitResponse`) the model charge is
    the response's recorded charge and the gains are profiled through it (Newton per sensor). With
    ``n_forward_draws >= 2`` the gains and the model charge are jackknifed over the draws
    (:func:`jackknife_profile`), which cancels the profiling's 1/n bias.
    """

    def __init__(
        self,
        forward,
        data,
        *,
        seed=0,
        q_floor_frac=0.01,
        gauge="log",
        n_forward_draws=1,
        jacobian_draws=8,
        data_var=None,
    ):
        data = jnp.asarray(data)
        if data.shape[0] != forward.S:
            raise ValueError(
                f"data has {data.shape[0]} rows but the forward produces "
                f"{forward.S} ({forward.W} wavelengths x {forward.n_sources} sources)"
            )
        self.forward = forward
        self.data = data
        self.data_sum = data.sum(0)
        self.q_floor = q_floor_frac * float(jnp.mean(data))
        self.gauge = gauge
        self.n_forward_draws = int(n_forward_draws)
        self.jacobian_draws = int(jacobian_draws)
        root = jax.random.PRNGKey(int(seed))
        self._forward_root = jax.random.fold_in(root, ROLE_FORWARD)
        self._jacobian_root = jax.random.fold_in(root, ROLE_JACOBIAN)
        self.n_configs = int(data.shape[0])
        self._J = None  # cached between refreshes
        self.response = forward.response
        self.data_var = None if data_var is None else jnp.asarray(data_var)
        self._wl_rows = (
            None
            if self.response is None
            else (jnp.arange(data.shape[0]) // forward.n_sources)[:, None]
        )
        # The Jacobian uses its data argument only as the weight's variance, sqrt(clip(., floor)).
        # Handing it the variance in use makes J weight exactly as r does.
        self._q_jac = (
            data if self.data_var is None else neyman_variance(data, self.q_floor, self.data_var)
        )

    def forward_key(self, step):
        """The residual's key on step ``step``; its draws fold in below it (:meth:`average`)."""
        return jax.random.fold_in(self._forward_root, step)

    def jacobian_key(self, step):
        """The Jacobian's key on step ``step``, on its own stream."""
        return jax.random.fold_in(self._jacobian_root, step)

    def predict(self, theta, step=0):
        """Model charge at unit gain, ``(S, n_sensors)``, on step ``step``'s draw."""
        return self.forward.average(
            theta, self.forward_key(step), jnp.ones(self.data.shape[1]), self.n_forward_draws
        )

    def _profiled_model(self, mu):
        """-> ``(k, model charge)``: the profiled gains and the model charge compared with the data."""
        if self.response is None:
            k = profile_gains(mu.sum(0) + 1e-12, self.data_sum, gauge=self.gauge)
            return k, k[None, :] * mu
        k = self.response.profile_gains(mu, self.data_sum, self._wl_rows, gauge=self.gauge)
        return k, self.response.model_charge(mu, k, self._wl_rows)

    def gains_and_model(self, theta, step=0):
        """-> ``(k, model charge)`` at ``theta`` on step ``step``'s forward draws.

        One draw: the gains profiled on it. Two or more: their jackknife
        (:func:`jackknife_profile`), which removes the profiling's 1/n bias.
        """
        if self.n_forward_draws < 2:
            return self._profiled_model(self.predict(theta, step))
        draws = self.forward.draws(
            theta, self.forward_key(step), jnp.ones(self.data.shape[1]), self.n_forward_draws
        )
        return jackknife_profile(draws, self.data_sum, self.gauge, self.response, self._wl_rows)

    def gains(self, theta, step=0):
        """The profiled per-PMT gains at ``theta``: the per-sensor QE/gain map, ``(n_sensors,)``."""
        return self.gains_and_model(theta, step)[0]

    def residuals(self, theta, step=0):
        """The Neyman residual, ``(S, n_sensors)``."""
        _, model = self.gains_and_model(theta, step)
        return neyman_residual(model, self.data, self.q_floor, self.data_var)

    def _jacobian(self, theta, lk, step):
        return self.forward.jacobian(
            theta,
            lk,
            self.jacobian_key(step),
            self._q_jac,
            self.q_floor,
            draws=range(self.jacobian_draws),
        )

    def jacobian(self, theta, step=0):
        """``∂r/∂θ``, ``(S, n_sensors, P)``, on the Jacobian's own stream; ``∂k/∂θ`` is dropped
        (:meth:`CalibrationForward.jacobian`)."""
        return self._jacobian(theta, jnp.log(self.gains(theta, step)), step)

    def loss(self, theta, step=0):
        """``Σr² / S``."""
        return float(jnp.sum(self.residuals(theta, step) ** 2) / self.n_configs)

    def grad_metric_loss(self, theta, step, refresh=True):
        """``(Jᵀr/S, JᵀJ/S, loss)``, the seam :func:`lucid.fitting.minimize.minimize` steps on.

        ``J`` is rebuilt only when ``refresh`` and is otherwise the cached one, so on a
        non-refresh step the search direction is stale too, not only its metric.
        """
        k, model = self.gains_and_model(theta, step)
        r = neyman_residual(model, self.data, self.q_floor, self.data_var)

        if refresh or self._J is None:
            self._J = self._jacobian(theta, jnp.log(k), step)
        J = self._J
        n = self.n_configs
        g = jnp.einsum("gnp,gn->p", J, r) / n
        H = jnp.einsum("gnp,gnq->pq", J, J) / n
        loss = float(jnp.sum(r**2) / n)
        return g, H, loss

    def nuisance(self, theta, step=0):
        """The profiled nuisance at ``theta``: ``{'gains': (n_sensors,)}``."""
        return {"gains": self.gains(theta, step)}


@functools.partial(jax.jit, static_argnums=(1, 2))
def _fold_grid(key, n, m):
    """``out[i, j] = fold_in(fold_in(key, i), j)``, ``(n, m)`` keys in one dispatch."""
    fold = jax.vmap(jax.random.fold_in, in_axes=(None, 0))
    return jax.vmap(lambda k: fold(k, jnp.arange(m)))(fold(key, jnp.arange(n)))


def _group_by_structure(sources):
    """-> [(indices, stacked_pytree_or_None)], sources grouped by PYTREE STRUCTURE.

    Groups appear in order of first appearance and indices are ascending within a group, so a
    homogeneous tail groups as one `sources[1:]` stack and the published layout resolves to
    [([0], None), ([1..7], stack)].

    Grouping by structure rather than by type is what admits an arbitrary layout: two sources can
    be mapped together precisely when they can be stacked, which is a property of their pytree,
    not of their class.
    """
    groups = []  # [(treedef, [indices])]
    for i, src in enumerate(sources):
        td = jax.tree_util.tree_structure(src)
        for g_td, ix in groups:
            if g_td == td:
                ix.append(i)
                break
        else:
            groups.append((td, [i]))
    out = []
    for _, ix in groups:
        stack = (
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[sources[i] for i in ix])
            if len(ix) > 1
            else None
        )
        out.append((ix, stack))
    return out


def _serial_map(fn, in_axes, n):
    """Serial stand-in for a mapped call: same per-item computation, results stacked.

    Numerically identical to a ``pmap``/``vmap`` over the leading axis for this use, and the only
    path available when fewer devices are present than sources.
    """
    jf = jax.jit(fn)

    def serial(*args):
        outs = []
        for i in range(n):
            a = [
                jax.tree_util.tree_map(lambda x: x[i], arg) if ax == 0 else arg
                for arg, ax in zip(args, in_axes)
            ]
            outs.append(jf(*a))
        return jnp.stack(outs)

    return serial
