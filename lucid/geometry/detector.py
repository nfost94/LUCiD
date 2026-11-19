"""
Functions for creating detectors from configuration files.
"""

import json
import os

from .box import Box  # noqa: F401

# Import subclasses so their @register_detector decorators run
from .cylinder import Cylinder  # noqa: F401
from .registry import get_detector_class
from .sphere import Sphere  # noqa: F401
from .string import StringTelescope  # noqa: F401  (telescope / volume detectors)


def load_detector_config(file_path):
    """
    Load detector configuration from JSON file.

    Parameters
    ----------
    file_path : str
        Path to the detector configuration JSON file

    Returns
    -------
    dict
        Detector configuration dictionary

    Raises
    ------
    ValueError
        If required fields are missing from the configuration
    """
    with open(file_path, "r") as file:
        config = json.load(file)

    # Validate required fields
    if "detector_type" not in config:
        raise ValueError(f"Detector config {file_path} missing required field 'detector_type'")

    if "material" not in config:
        raise ValueError(
            f"Detector config {file_path} missing required field 'material'.\n"
            f'Please add \'"material": "water"\' to the configuration file.'
        )

    if "geometry_definitions" not in config:
        raise ValueError(
            f"Detector config {file_path} missing required field 'geometry_definitions'"
        )

    return config


# PMT types a surface detector can carry (``sensor_shape``):
#   'sphere'   the wall-mounted hemisphere of the config's sensor_radius (the default)
#   'hk50cm'   the HK 50 cm Box&Line, a cap of a sphere (lucid.propagation.hk_pmt_geometry)
SENSOR_SHAPES = ("sphere", "hk50cm")


def resolve_sensor_shape(file_path, sensor_shape=None):
    """The PMT type for a detector config: ``sensor_shape`` given, else the config's, else 'sphere'.

    A detector config names its PMT type as ``sensor_shape`` in ``geometry_definitions``; the type
    then sets the PMT's geometry. A ``sensor_shape`` given here is for configs that name none; one
    that disagrees with the config's raises.
    """
    configured = load_detector_config(file_path).get("geometry_definitions", {}).get("sensor_shape")
    if sensor_shape is not None and configured is not None and sensor_shape != configured:
        raise ValueError(
            f"sensor_shape={sensor_shape!r} disagrees with {file_path!r}, whose PMT "
            f"type is {configured!r}"
        )
    shape = sensor_shape if sensor_shape is not None else (configured or "sphere")
    if shape not in SENSOR_SHAPES:
        raise ValueError(f"sensor_shape must be one of {SENSOR_SHAPES}, got {shape!r}")
    return shape


def generate_detector(file_path):
    """Generate a detector from a JSON config file using the geometry registry.

    The config's ``detector_type`` (e.g. 'cylinder', 'sphere', 'box', 'string') is looked up in
    the registry, and that class's ``from_config`` builds the detector from
    ``geometry_definitions``. Relative file paths in it (``npz_file_path``) resolve against the
    config file's directory.

    The PMT type (``sensor_shape`` in ``geometry_definitions``) is not read here: it sets the
    sensor's shape, not where sensors sit. See :func:`resolve_sensor_shape`.
    """
    config = load_detector_config(file_path)
    cls = get_detector_class(config["detector_type"])
    config_dir = os.path.dirname(os.path.abspath(file_path))
    return cls.from_config(config["geometry_definitions"], config_dir)
