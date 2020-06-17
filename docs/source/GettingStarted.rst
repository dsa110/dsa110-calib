================
Getting Started
================

Installation 
------------
To get started, clone dsacalib from the github repository `https://github.com/dsa110/dsa110-calib <https://github.com/dsa110/dsa110-calib>`_ and follow the installation directions in the ReadMe.  dsacalib is built on CASA - the required modules can be downloaded using pip: 

.. code-block:: bash

   pip install casatools --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple --no-cache-dir
   pip install casatasks --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple --no-cache-dir
   pip install casadata --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple --no-cache-dir

Real-time pipelines
-------------------
If you are using dsacalib for calibration of DSA antenna gains and delays, you likely want to use one of the pre-built pipelines.  These piplines include integration with the greater DSA system through writing of logs to syslog and pushing solutions to etcd.  These pipelines are in `dsacalib/scripts`.  The currently-available pipelines are:

* `triple_antenna_corr` -  Reads in the output FITS files of the 3-antenna correlator and uses CASA to perform baseline-based calibration.  Antenna solutions are calculated from the baseline solutions and pushed to etcd

If you are looking to build your own pipeline, these may be a good place to start.