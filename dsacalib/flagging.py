"""Flag visibilities in measurement sets."""

import numpy as np
from casatasks import flagdata
import casatools as cc

def flag_antenna(msname, antenna, datacolumn="data", pol=None):
    """Flags an antenna in a measurement set using CASA.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    antenna : str
        The antenna to flag. If type *str*, this is the name of the antenna. If
        type *int*, the index of the antenna in the measurement set.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    pol : str
        The polarization to flag. Must be `'A'` (which is mapped to
        polarization 'XX' of the CASA measurement set) or `'B'` (mapped to
        polarization 'YY').  Can also be `None`, for which both polarizations
        are flagged.  Defaults to `None`.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    if isinstance(antenna, int):
        antenna = str(antenna)
    error = 0
    ag = cc.agentflagger()
    error += not ag.open(f"{msname}.ms")
    error += not ag.selectdata()
    rec = {}
    rec["mode"] = "manual"
    rec["datacolumn"] = datacolumn
    rec["antenna"] = antenna
    if pol is not None:
        rec["correlation"] = "XX" if pol == "A" else "YY"
    else:
        rec["correlation"] = "XX,YY"
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error


def flag_manual(msname, key, value, datacolumn="data", pol=None):
    """Flags a measurement set in CASA using a flagging string.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    key : str
        The CASA-interpreted flagging specifier.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    pol : str
        The polarization to flag. Must be `'A'` (which is mapped to
        polarization 'XX' of the CASA measurement set) or `'B'` (mapped to
        polarization 'YY').  Can also be `None`, for which both polarizations
        are flagged.  Defaults to `None`.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open(f"{msname}.ms".format(msname))
    error += not ag.selectdata()
    rec = {}
    rec["mode"] = "manual"
    rec["datacolumn"] = datacolumn
    rec[key] = value
    if pol is not None:
        rec["correlation"] = "XX" if pol == "A" else "YY"
    else:
        rec["correlation"] = "XX,YY"
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error


def flag_baselines(msname, datacolumn="data", uvrange="2~50m"):
    """Flags an antenna in a measurement set using CASA.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    uvrange : str
        The uvrange to flag. Should be CASA-interpretable.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open(f"{msname}.ms")
    error += not ag.selectdata()
    rec = {}
    rec["mode"] = "manual"
    rec["datacolumn"] = datacolumn
    rec["uvrange"] = uvrange
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error


def reset_flags(msname, datacolumn=None):
    """Resets all flags in a measurement set, so that all data is unflagged.

    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open(f"{msname}.ms")
    error += not ag.selectdata()
    rec = {}
    rec["mode"] = "unflag"
    if datacolumn is not None:
        rec["datacolumn"] = datacolumn
    rec["antenna"] = ""
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error


def reset_all_flags(msname):
    """Reset all flags in a measurement set."""
    dc.reset_flags(msname, datacolumn="data")
    dc.reset_flags(msname, datacolumn="model")
    dc.reset_flags(msname, datacolumn="corrected")


def flag_zeros(msname, datacolumn="data"):
    """Flags all zeros in a measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open(f"{msname}.ms")
    error += not ag.selectdata()
    rec = {}
    rec["mode"] = "clip"
    rec["clipzeros"] = True
    rec["datacolumn"] = datacolumn
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error


# TODO: Change times to not use mjds, but mjd instead
def flag_badtimes(msname, times, bad, nant, datacolumn="data", verbose=False):
    """Flags bad time bins for each antenna in a measurement set using CASA.

    Could use some work to select antennas to flag in a smarter way.

    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    times : ndarray
        A 1-D array of times, type float, seconds since MJD=0. Times should be
        equally spaced and cover the entire time range of the measurement set,
        but can be coarser than the resolution of the measurement set.
    bad : ndarray
        A 1-D boolean array with dimensions (len(`times`), `nant`). Should have
        a value of ``True`` if the corresponding timebins should be flagged.
    nant : int
        The number of antennas in the measurement set (includes ones not in the
        visibilities).
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    verbose : boolean
        If ``True``, will print information about the antenna/time pairs being
        flagged.  Defaults to ``False``.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    tdiff = np.median(np.diff(times))
    ag = cc.agentflagger()
    error += not ag.open(f"{msname}.ms".format(msname))
    error += not ag.selectdata()
    for i in range(nant):
        rec = {}
        rec["mode"] = "clip"
        rec["clipoutside"] = False
        rec["datacolumn"] = datacolumn
        rec["antenna"] = str(i)
        rec["polarization_type"] = "XX"
        tstr = ""
        for j, timesj in enumerate(times):
            if bad[j]:
                if len(tstr) > 0:
                    tstr += "; "
                tstr += f"{timesj - tdiff / 2}~{timesj + tdiff / 2}"
        if verbose:
            print(f"For antenna {i}, flagged: {tstr}")
        error += not ag.parseagentparameters(rec)
        error += not ag.init()
        error += not ag.run()

        rec["polarization_type"] = "YY"
        tstr = ""
        for j, timesj in enumerate(times):
            if bad[j]:
                if len(tstr) > 0:
                    tstr += "; "
                tstr += f"{timesj - tdiff / 2}~{timesj + tdiff / 2}"
        if verbose:
            print(f"For antenna {i}, flagged: {tstr}")
        error += not ag.parseagentparameters(rec)
        error += not ag.init()
        error += not ag.run()
    error += not ag.done()

    return error


def flag_rfi(msname: str) -> None:
    """Flag RFI using tfcrop in casa."""
    flagdata(f"{msname}.ms", mode="tfcrop")


def flag_antennas_using_delays(
        antenna_delays: np.ndarray, kcorr: np.ndarray, msname: str, kcorr_thresh: float=0.3,
        logger: "DsaSyslogger"=None
):
    """Flags antennas by comparing the delay on short times to the delay cal.

    Parameters
    ----------
    antenna_delays : ndarray
        The antenna delays from the 2kcal calibration file, calculated on short
        timescales.
    kcorr : ndarray
        The antenna delays from the kcal calibration file, calculated over the
        entire calibration pass.
    msname : str
        The path to the measurement set. Will open `msname`.ms
    kcorr_thresh : float
        The tolerance for descrepancies between the antenna_delays and kcorr,
        in nanoseconds.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    """
    error = 0
    percent_bad = (np.abs(antenna_delays - kcorr) > 1).sum(1).squeeze(1).squeeze(
        1
    ) / antenna_delays.shape[1]
    for i in range(percent_bad.shape[0]):
        for j in range(percent_bad.shape[1]):
            if percent_bad[i, j] > kcorr_thresh:
                error += not flag_antenna(
                    msname, str(i + 1), pol="B" if j else "A"
                )
                message = f"Flagged antenna {i + 1}{'B' if j else 'A'} in {msname}"
                if logger is not None:
                    logger.info(message)
                else:
                    print(message)
    return error
