"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import shutil
import os
import glob
from typing import List, Union
from pathlib import Path

import h5py
import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.time import Time
import pandas

import scipy  # pylint: disable=unused-import
from casacore.tables import table
import dsautils.calstatus as cs
from dsautils.dsa_syslog import DsaSyslogger

from dsacalib.calibrator_observation import CalibratorObservation
import dsacalib.utils as du


class PipelineComponent:
    """A component of the real-time pipeline."""

    def __init__(self, logger: DsaSyslogger, throw_exceptions: bool):
        self.description = 'Pipeline component'
        self.error_code = 0
        self.nonfatal_error_code = 0
        self.logger = logger
        self.throw_exceptions = throw_exceptions

    def target(self):
        return 0

    def __call__(self, status: int, *args) -> int:
        """Handle fatal and nonfatal errors.

        Return an updated status that reflects any errors that occurred in `target`.
        """
        try:
            error = self.target(*args)

        except Exception as exc:
            status = cs.update(status, self.error_code)
            du.exception_logger(self.logger, self.description, exc, self.throw_exceptions)

        else:
            if error > 0:
                status = cs.update(status, self.nonfatal_error_code)
                message = f'Non-fatal error occured in {self.description}'
                du.warning_logger(self.logger, message)

        return status


class Flagger(PipelineComponent):
    """The ms flagger.  Flags baselines, zeros, bad antennas, and rfi."""

    def __init__(self, logger: DsaSyslogger, throw_exceptions: bool):
        """Describe Flagger and error code if fails."""
        super().__init__(logger, throw_exceptions)
        self.description = 'flagging'
        self.error_code = (
            cs.FLAGGING_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME)
        self.nonfatal_error_code = cs.FLAGGING_ERR

    def target(self, calobs: CalibratorObservation):
        """Flag data in the measurement set."""
        error = calobs.set_flags()
        return error


class DelayCalibrater(PipelineComponent):
    def __init__(self, logger: DsaSyslogger, throw_exceptions: bool):
        super().__init__(logger, throw_exceptions)
        self.description = 'delay calibration'
        self.error_code = (
            cs.DELAY_CAL_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME)
        self.nonfatal_error_code = cs.DELAY_CAL_ERR

    def target(self, calobs):
        error = calobs.delay_calibration()
        return error


class BandpassGainCalibrater(PipelineComponent):
    def __init__(self, logger: DsaSyslogger, throw_exceptions: bool):
        super().__init__(logger, throw_exceptions)
        self.description = 'bandpass and gain calibration'
        self.error_code = (
            cs.GAIN_BP_CAL_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_GAINCALTIME)
        self.nonfatal_error_code = cs.GAIN_BP_CAL_ERR

    def target(self, calobs):
        if not calobs.config['delay_bandpass_table_prefix']:
            error = calobs.bandpass_calibration()
            error += calobs.gain_calibration()
            error += calobs.bandpass_calibration()

        else:
            error += calobs.gain_calibration()

        return error


class BFWeightCreater(PipelineComponent):
    def __init__(self, logger: "DsaSyslogger", throw_exceptions: bool):
        super().__init__(logger, throw_exceptions)
        self.description = 'beamformer weight creation'

    def target(self, calobs):
        calobs.create_beamformer_weights()
        return 0


def calibrate_measurement_set(
        msname: str, cal: CalibratorSource, scan: Scan = None,
        logger: DsaSyslogger = None, throw_exceptions: bool = False, **kwargs) -> int:
    calobs = CalibratorObservation(msname, cal, scan)
    calobs.set_calibration_parameters(**kwargs)
    flag = Flagger(logger, throw_exceptions)
    delaycal = DelayCalibrater(logger, throw_exceptions)
    bpgaincal = BandpassGainCalibrater(logger, throw_exceptions)
    bfgen = BFWeightCreater(logger, throw_exceptions)

    message = f"Beginning calibration of {msname}."
    logger.info(message)

    status = 0
    calobs.reset_calibration()

    if not calobs.config['reuse_flags']:
        status |= flag(calobs)

    if not calobs.config['delay_bandpass_table_prefix']:
        status |= delaycal(calobs)

    status |= bpgaincal(calobs)

    combine_tables(msname, f"{msname}_{cal.name}", calobs.config['delay_bandpass_table_prefix'])
    status |= bfgen(calobs)

    message = f"Completed calibration of {msname} with status {cs.decode(status)}"
    logger.info(message)

    return status

