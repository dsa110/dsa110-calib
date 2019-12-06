"""DSA_CONSTANTS.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Constants used for the DSA analysis
"""
import astropy.units as u
import astropy.constants as c
import numpy as np
import dsacalib

# The number of seconds in a sidereal day
seconds_per_sidereal_day = 3600*23.9344699

# The number of seconds in a day
seconds_per_day = 3600*24

# Time between time the packet says as start and first
# sample recorded
time_offset = 4.294967296
casa_time_offset = 0.00042824074625968933 # in days

# The integration time of the visibilities in seconds
tsamp = 8.192e-6*128*384

# The longitude and latitude of the OVRO site
# in radians
ovro_lon = (-118.283400*u.deg).to_value(u.rad)
ovro_lat = (37.233386*u.deg).to_value(u.rad)

# c expressed in units relevant to us
c_GHz_m = c.c.to_value(u.GHz*u.m)

# pointing declination of the array in radians
pt_dec = '+73d40m0s'

# Amount to integrate data by when fringestopping
# Currently integrating for 10-s 
# When commissioning DSA-110, want 1-s integrations
nint = int(np.floor(10/tsamp))

# Default sky model parameters
f0 = 1.4        # Frequency of flux in GHz
spec_idx = -0.7 # Spectral index 

# Backup IERS table
iers_table    = 'file://{0}/data/finals2000A.all'.format(dsacalib.__path__[0])
# Templates & other package data
pkg_data_path = '{0}/data/'.format(dsacalib.__path__[0])