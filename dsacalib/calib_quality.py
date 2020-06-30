""""Routines to automate quality verification of CASA calibration solutions.

"""

# Things we may want to consider including:
# - when flagging data, checking how much data is flagged and raising a warning if 
#   it's over 20% or so
# - when searching for delay, bandpass, amplitude gain, phase gain solutions, 
#   checking that casa found solutions on the majority of the intervals
# - after delay solutions, checking that:
#   - the median of the solution on short timescales is close (< 1 ns) to the 
#     solution on long timescales
#   - the solutions for each antenna on short timescales are flat 
#   - less than 20% of the solutions on short timescales are outside of 
#     the 1.5 ns cutoff (this should already be taken care of in the 
#     data flagging)
# - after bandpass solutions, checking that:
#   - bandpass flat for each antenna
# - after phase calibration, checking that:
#   - phase solutions on a short timescale don't differ substantially from 
#     the phase solutions on long timescales
# - after gain calibration, checking that:
#   - gain solutions smooth

