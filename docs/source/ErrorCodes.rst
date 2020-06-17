Error Codes for Real-time Calibration
======================================

The dsacalib real-time pipelines work as part of the larger software system of the DSA by:

* Writing messages to syslog, instead of warning and errors to stderr
* Pushing real-time calibration delays and gains to etcd

Add some notes on pyutils, etc.

If an error occurs during calibration, the dsacalib pipeline will return a non-zero error code instead of a traceback, while details on the error will be written to syslog.  Bits 0-3 code the error description, while the remaining bits each indicate the validity of information passed to etcd.  If you are building a new pipeline for the DSA, use these same error codes with syslog and etcd integration.

Bits 0-3: Error description
---------------------------

Bits 0-3 code a single value describing the source of the error.  
Values of 1 through 5 indicate the error occured in reading the visibilities or fringestopping.  
Values of 6 through 8 indicate the error occured in calibration.  
Values of 9 through 14 indicate the error occured in storage of the calibration solutions in etcd.  
Value of 15 indicates that some other error occured.

The detailed descriptions for each code are:

* 1 = HDF5 or FITS file containing the visibilities cannot be found or opened
* 2 = HDF5 of FITS file doesn't contain the calibration source
* 3 = File doesn't meet specifications or is otherwise correct.  This error is returned, for e.g., if the file doesn't contain the specified number of antennas (which is a required input for reading FITS files produced by the 3- and 6-antenna correlators), the reference antenna is not in the file, the dimensions of the visibilities don't match the times, frequencies, or polarizations specified in the file (or input), etc.
* 4 = Fringestopping on the calibrator source failed
* 5 = Writing the fringe-stopped measurement set failed 
* 6 = Failure in flagging 
* 7 = Delay calibration failed
* 8 = Gain or bandpass calibration failed
* 9 = caltable_to_etcd cannot open the antenna gain table
* 10 = caltable_to_etcd cannot open antenna delay table
* 11 = caltable_to_etcd was passed an invalid calname
* 12 = caltable_to_etcd was passed an invalid antenna_order
* 13 = caltable_to_etcd was passed an invalid calibration time
* 14 = caltable_to_etcd was passed an invalid status
* 15 = An error not in this list occured 

Failures in delay and gain calibration gain typically occur because tables containing previous solutions cannot be opened, too much data in the measurement set is flagged, the data doesn't have high enough signal-to-noise, or (in the case of 3-antenna solutions) CASA returned baseline-based solutions, but calculation of the antenna-based solutions from these failed.

Bits 4 - 15: Validity of solutions passed to etcd
--------------------------------------------------
Bits 4-15 each correspond to a particular piece of information passed to etcd.  A value of 0 indicates the information is valid, 1 indicates the information is invalid.

* Bit 4 = ant_num (antenna number)
* Bit 5 = time (calibration time, corresponds to the meridian-crossing time of the calibrator source
* Bit 6 = pol 
* Bit 7 = gainamp (gain amplitude)
* Bit 8 = gainphase (gain phase)
* Bit 9 = delay (antenna delay)
* Bit 10 = calsource (calibrator source)
* Bit 11 = gaincaltime_offset (offset of gain calibration time given by CASA and the meridian crossing time)
* Bit 12 = delaycaltime_offset (offset of delay calibration time given by CASA and the meridian crossing time)
* Bit 13 = sim (if the data is from a simulation or valid data)
* Bit 14 = status (this error code)
* Bit 15 is currently unused

For details on the etcd format, see (insert link and copy information).