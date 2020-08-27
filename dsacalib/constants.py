"""Constants used in the calibration of DSA-110 visibilities.

Author: Dana Simard, dana.simard@astro.caltech.edu, 10/2019
"""
import astropy.units as u
import astropy.constants as c
import numpy as np
import dsacalib
import os.path

# The number of seconds in a sidereal day
SECONDS_PER_SIDEREAL_DAY = 3600*23.9344699

# The number of seconds in a day
SECONDS_PER_DAY = 3600*24

DEG_PER_HOUR = 360/SECONDS_PER_SIDEREAL_DAY*3600

# Time between time the packet says as start and first
# sample recorded
TIME_OFFSET = 4.294967296
CASA_TIME_OFFSET = 0.00042824074625968933 # in days

# The integration time of the visibilities in seconds
TSAMP = 8.192e-6*128*384

# The longitude and latitude of the OVRO site
# in radians
OVRO_LON = (-118.283400*u.deg).to_value(u.rad)
OVRO_LAT = (37.233386*u.deg).to_value(u.rad)

# c expressed in units relevant to us
C_GHZ_M = c.c.to_value(u.GHz*u.m)

# Amount to integrate data by when fringestopping
# Currently integrating for 10-s
# When commissioning DSA-110, want 1-s integrations
NINT = int(np.floor(10/TSAMP))

# Default sky model parameters
F0 = 1.4        # Frequency of flux in GHz
SPEC_IDX = -0.7 # Spectral index

# Backup IERS table
IERS_TABLE = '{0}/data/finals2000A.all'.format(dsacalib.__path__[0])
# Templates & other package data
PKG_DATA_PATH = '{0}/data/'.format(dsacalib.__path__[0])

