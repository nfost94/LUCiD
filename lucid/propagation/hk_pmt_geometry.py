"""Geometry of the HK 50 cm Box&Line PMT, WCSim's BoxandLine20inchHQE.

Lengths are in metres. The PMT is the cap of a sphere standing on the wall: an aperture of radius
APERTURE_RADIUS in the wall plane and a dome EXPOSE_HEIGHT high, as WCSim builds it
(WCSimPMTObject.cc, BoxandLine20inchHQE::GetRadius and ::GetExposeHeight). The sphere it is cut
from, and how far that sphere's centre sits behind the wall, follow from the two (see
`lucid.overlap.cap_sphere`). This module contains geometry definitions only.
"""
from lucid.overlap import cap_sphere

APERTURE_RADIUS = 0.254
EXPOSE_HEIGHT = 0.18
SPHERE_RADIUS, SPHERE_CENTRE_DEPTH = cap_sphere(APERTURE_RADIUS, EXPOSE_HEIGHT)
