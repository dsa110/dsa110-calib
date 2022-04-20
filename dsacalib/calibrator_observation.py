"""Calibration routines for the DSA-110 with CASA."""
import os
import shutil

import dsautils.cnf as dsc

import dsacalib.calib as dc
import dsacalib.flagging as df
import dsacalib.plotting as dp


class CalibratorObservation:
    """A calibrator observation used to obtain beamformer and voltage calibration solutions."""

    def __init__(self, msname: str, cal: "dsacalib.utils.src") -> None:
        """Initialize the calibrator observation, including settings for calibration.

        `msname` should exclude the ".ms" extension
        """
        self.msname = msname
        self.cal = cal
        self.table_prefix = f"{self.msname}_{self.cal.name}"
        self.config = get_configuration()

    def set_calibration_parameters(self, **kwargs) -> None:
        """Update default settings for calibration."""
        for key, arg in kwargs.items():
            if key not in self.config:
                raise RuntimeError(
                    f"{key} not in calibration_settings\n"
                    f"Allowed keys are {list(self.config.keys())}")
            self.config[key] = arg

        if isinstance(self.config["refants"], (int, str)):
            self.config["refants"] = list(self.config["refants"])

    def reset_calibration(self) -> None:
        """Remove existing calibration tables."""
        tables_to_remove = [
            f"{self.table_prefix}_{ext}" for ext in [
                "2kcal", "kcal", "bkcal", "gacal", "gpcal", "bcal",]]
        if self.config["forsystemhealth"]:
            tables_to_remove += [f"{self.table_prefix}_2gcal"]

        for path in tables_to_remove:
            if os.path.exists(path):
                shutil.rmtree(path)

    def set_flags(self) -> int:
        """Reset flags and set new flags."""
        df.reset_all_flags(self.msname)
        error = 0
        if self.config["bad_uvrange"]:
            error += df.flag_baselines(self.msname, uvrange=self.config["bad_uvrange"])
        error += df.flag_zeros(self.msname)

        for ant in self.config["bad_antennas"]:
            error += df.flag_antenna(self.msname, ant)

        for entry in self.config["manual_flags"]:
            error += df.flag_manual(self.msname, entry[0], entry[1])

        df.flag_rfi(self.msname)

        return error

    def delay_calibration(self, t2: str = "60s") -> int:
        """Calibrate delays."""
        error = 0

        # Delay calibration on two timescales
        error += dc.delay_calibration(
            self.msname, self.cal.name, refants=self.config["refants"], t2=t2)
        _check_path(f"{self.table_prefix}_kcal")

        # Compare delay calibration on the timescales to flag extra antennas
        _times, antenna_delays, kcorr, _ant_nos = dp.plot_antenna_delays(
            self.msname, self.cal.name, show=False)
        error += df.flag_antennas_using_delays(antenna_delays, kcorr, self.msname)
        try:
            _check_path(f"{self.table_prefix}_2kcal")
        except AssertionError:
            error += 1

        # Delay calibration again on one or two timescales
        for ext in ["kcal", "2kcal"]:
            shutil.rmtree(f"{self.table_prefix}_{ext}")
        error += dc.delay_calibration(
            self.msname, self.cal.name, refants=self.config["refants"],
            t2=t2 if self.config["forsystemhealth"] else None)
        _check_path(f"{self.table_prefix}_kcal")

        return error

    def bandpass_and_gain_cal(self, tbeam: str = "60s") -> int:
        """Gain and bandpass calibration."""
        error = dc.gain_calibration(
            self.msname,
            self.cal.name,
            self.config['refants'][0],
            blbased=False,
            forsystemhealth=self.config["forsystemhealth"],
            keepdelays=self.config["keepdelays"],
            interp_thresh=1.5,
            interp_polyorder=7,
            tbeam=tbeam)
        print(error)
        dc.combine_bandpass_and_delay(self.table_prefix, self.config["forsystemhealth"])

        return error
    
    def quick_delay_calibration(self) -> int:
        error = 0
        error += dc.delay_calibration(
            self.msname, self.cal.name, refants=self.config["refants"])
        _check_path(f"{self.table_prefix}_kcal")
        return error

def get_configuration() -> dict:
    """Get the default configuration for calibration."""
    dsaconf = dsc.Conf()
    cal_params = dsaconf.get("cal")

    config = {
        "refants": cal_params["refant"],

        "bad_antennas" : [],
        "bad_uvrange" : "2~50m",
        "manual_flags": [],

        "forsystemhealth": False,
        "keepdelays": False,

        "reuse_flags": False}

    return config


def _check_path(fname: str) -> None:
    """Raises an AssertionError if the path `fname` does not exist.

    Parameters
    ----------
    fname : str
        The file to check existence of.
    """
    assert os.path.exists(fname), f"File {fname} does not exist"
