"""DSA_CONSTANTS.PY
Dana Simard, dana.simard@astro.caltech.edu, 10/2019
Constants used for the DSA analysis
"""
import astropy.units as u

# The number of seconds in a sidereal day
seconds_per_sidereal_day = 3600*23.9344699

# The number of seconds in a day
seconds_per_day = 3600*24

# An unknown parameter - clock offset in seconds of DSA?
time_offset = 4.294967296

# The integration time of the visibilities
tsamp = 8.192e-6*128*384

# The longitude and latitude of the OVRO site
ovro_lon = -118.283400*u.deg
ovro_lat = 37.233386*u.deg