"""Functions for calibration of DSA-110 visibilities.

These functions use the CASA package casatools to calibrate
visibilities stored as measurement sets.

Author: Dana Simard, dana.simard@astro.caltech.edu, 10/2019

"""
import os
import shutil
from typing import List
from copy import deepcopy

# Always import scipy before casatools
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import medfilt

import numpy as np
import casatools as cc
from casacore.tables import table, tablecopy

from dsacalib.ms_io import read_caltable


def delay_calibration_worker(
        msname: str, sourcename: str, refant: str, t: str, combine_spw: bool, name: str
) -> int:
    r"""Calibrates delays using CASA.

    Uses CASA to calibrate delays and write the calibrated visibilities to the
    corrected_data column of the measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The measurement set `msname`.ms will
        be opened.
    sourcename : str
        The name of the calibrator source. The calibration table will be
        written to `msname`\_`sourcename`\_kcal.
    refant : str
        The reference antenna to use in calibration. If  type *str*, this is
        the name of the antenna.  If type *int*, it is the index of the antenna
        in the measurement set.
    t : str
        The integration time to use before calibrating, e.g. ``'inf'`` or
        ``'60s'``.  See the CASA documentation for more examples. Defaults to
        ``'inf'`` (averaging over the entire observation time).
    combine_spw : boolean
        If True, distinct spws in the same ms will be combined before delay
        calibration.
    name : str
        The suffix for the calibration table.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combine_spw:
        combine = "field,scan,obs,spw"
    else:
        combine = "field,scan,obs"
    error = 0
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += not cb.setsolve(
        type="K",
        t=t,
        refant=refant,
        combine=combine,
        table=f"{msname}_{sourcename}_{name}",
    )
    error += not cb.solve()
    error += not cb.close()
    return error


def delay_calibration(
        msname: str, sourcename: str, refants: List[str], t1: str = "inf", t2: str = None,
        combine_spw: bool = False
) -> int:
    r"""Calibrates delays using CASA.

    Uses CASA to calibrate delays and write the calibrated visibilities to the
    corrected_data column of the measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The measurement set `msname`.ms will
        be opened.
    sourcename : str
        The name of the calibrator source. The calibration table will be
        written to `msname`\_`sourcename`\_kcal.
    refants : list(str)
        The reference antennas to use in calibration. If list items are type
        *str*, this is the name of the antenna.  If type *int*, it is the index
        of the antenna in the measurement set. An average is done over all
        reference antennas to get the final delay table.
    t1 : str
        The integration time to use before calibrating to generate the final
        delay table, e.g. ``'inf'`` or ``'60s'``.  See the CASA documentation
        for more examples. Defaults to ``'inf'`` (averaging over the entire
        observation time).
    t2 : str
        The integration time to use before fast delay calibrations used to flag
        antennas with poor delay solutions.
    combine_spw : boolean
        If True, distinct spws in the same ms will be combined before delay
        calibration.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    # TODO : revisit and get this to work with a list of antennas
    assert isinstance(refants, list)
    error = 0
    refant = None
    tlist = [t1]
    if t2:
        tlist.append(t2)

    for t in tlist:
        kcorr = None
        for refant in refants:
            if isinstance(refant, str):
                refantidx = int(refant) - 1
            else:
                refantidx = refant
            error += delay_calibration_worker(
                msname,
                sourcename,
                refant,
                t,
                combine_spw,
                f"ref{refant}_{'' if t == t1 else '2'}kcal",
            )
            if kcorr is None:
                kcorr, _, flags, _, ant2 = read_caltable(
                    f"{msname}_{sourcename}_ref{refant}_{'' if t==t1 else '2'}kcal",
                    cparam=False,
                    reshape=False,
                )
            else:
                kcorrtmp, _, flagstmp, _, ant2tmp = read_caltable(
                    f"{msname}_{sourcename}_ref{refant}_{'' if t==t1 else '2'}kcal",
                    cparam=False,
                    reshape=False,
                )
                antflags = (
                    np.abs(flags.reshape(flags.shape[0], -1).mean(axis=1) - 1) < 1e-5
                )
                assert antflags[refantidx] == 0, (
                    f"Refant {refant} is flagged in kcorr! "
                    "Choose refants that are separated in uv-space."
                )
                kcorr[antflags, ...] = kcorrtmp[antflags, ...] - kcorr[refantidx, ...]
                ant2[antflags, ...] = ant2tmp[antflags, ...]
                flags[antflags, ...] = flagstmp[antflags, ...]

        # write out to a table
        with table(
                f"{msname}_{sourcename}_ref{refant}_{'' if t == t1 else 2}kcal",
                readonly=False,
        ) as tb:
            tb.putcol("FPARAM", kcorr)
            tb.putcol("FLAG", flags)
            tb.putcol("ANTENNA2", ant2)
        newpath = f"{msname}_{sourcename}_{'' if t==t1 else '2'}kcal"
        if os.path.exists(newpath):
            shutil.rmtree(newpath)
        os.rename(
            f"{msname}_{sourcename}_ref{refant}_{'' if t == t1 else '2'}kcal",
            newpath
        )
    return error


def gain_calibration(
        msname: str, sourcename: str, refant: str, blbased: bool = False,
        forsystemhealth: bool = False, keepdelays: bool = False, tbeam: str = "30s") -> int:
    r"""Use CASA to calculate bandpass and complex gain solutions.

    Saves solutions to calibration tables and calibrates the measurement set by
    applying delay, bandpass, and complex gain solutions.  Uses baseline-based
    calibration routines within CASA.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    sourcename : str
        The name of the calibrator source.  The calibration table will be
        written to `msname`\_`sourcename`\_kcal.
    refant : str
        The reference antenna to use in calibration.  If type *str*, this is
        the name of the antenna.  If type *int*, it is the index of the antenna
        in the measurement set.
    blbased : boolean
        Set to True if baseline-based calibration desired.
    forsystemhealth : boolean
        Set to True if gain calibration is for system health monitoring. Delays
        will be kept at full resolution. If set to False, then at least some of
        the delay will be incorporated into the bandpass gain table.
    keepdelays : boolean
        Set to True if you want to update the delays currently set in the
        system. In this case, delay changes of integer 2 ns will be kept in the
        delay calibration table, and residual delays will be incorporated into
        the bandpass gain table. If set to False, all of the delay will be
        incorporated into the bandpass gain table.
    tbeam : str
        The integration time to use when measuring gain variations over time,
        e.g. ``'inf'`` or ``'60s'``.  See the CASA documentation for more
        examples.
    interp_thresh : float
        Sets flagging of bandpass solutions before interpolating in order to
        smooth the solutions. After median baselining, any points that deviate
        by more than interp_thresh*std are flagged.
    interp_polyorder : int
        The order of the polynomial used to smooth bandpass solutions.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    combine = "field,scan,obs"
    spwmap = [-1]
    error = 0
    fref_snaps = 0.03  # SNAPs correct to freq of 30 MHz

    # Convert delay calibration into a bandpass representation
    caltables = [
        {
            "table": f"{msname}_{sourcename}_kcal",
            "type": "K",
            "spwmap": spwmap,
        }
    ]

    if not forsystemhealth:
        with table(f"{msname}.ms/SPECTRAL_WINDOW") as tb:
            fobs = np.array(tb.CHAN_FREQ[:]).squeeze(0) / 1e9
            fref = np.array(tb.REF_FREQUENCY[:]) / 1e9
        cb = cc.calibrater()
        error += not cb.open(f"{msname}.ms")
        error += apply_calibration_tables(cb, caltables)
        error += not cb.setsolve(
            type="MF" if blbased else "B",
            combine=combine,
            table=f"{msname}_{sourcename}_bkcal",
            refant=refant,
            apmode="a",
            solnorm=True,
        )
        error += not cb.solve()
        error += not cb.close()

        with table(f"{msname}_{sourcename}_kcal", readonly=False) as tb:
            kcorr = np.array(tb.FPARAM[:])
            tb.putcol("FPARAM", np.zeros(kcorr.shape, kcorr.dtype))

        with table(f"{msname}_{sourcename}_bkcal", readonly=False) as tb:
            bpass = np.array(tb.CPARAM[:])
            bpass = np.ones(bpass.shape, bpass.dtype)
            kcorr = kcorr.squeeze()
            bpass *= np.exp(
                2j
                * np.pi
                * (fobs[:, np.newaxis] - fref)
                * (kcorr[:, np.newaxis, :])  # -kcorr[int(refant)-1, :]
            )
            tb.putcol("CPARAM", bpass)
        caltables += [
            {
                "table": f"{msname}_{sourcename}_bkcal",
                "type": "B",
                "spwmap": spwmap,
            }
        ]

    error += solve_gain_calibration(
        msname, sourcename, refant, caltables, forsystemhealth, combine, spwmap,
        blbased, tbeam)

    if not forsystemhealth and keepdelays:
        with table(f"{msname}_{sourcename}_kcal", readonly=False) as tb:
            fparam = np.array(tb.FPARAM[:])
            newparam = np.round(kcorr[:, np.newaxis, :] / 2) * 2
            print("kcal", fparam.shape, newparam.shape)
            tb.putcol("FPARAM", newparam)
        with table(f"{msname}_{sourcename}_bkcal", readonly=False) as tb:
            bpass = np.array(tb.CPARAM[:])
            print(newparam.shape, bpass.shape, fobs.shape)
            bpass *= np.exp(-2j * np.pi * (fobs[:, np.newaxis] - fref_snaps) * newparam)
            print(bpass.shape)
            tb.putcol("CPARAM", bpass)

    return error

def solve_gain_calibration(
        msname: str, sourcename: str, refant: str, caltables: List[dict],
        forsystemhealth: bool, combine: str = "field,scan,obs", spwmap: List = None,
        blbased: bool = False, tbeam: str = "60s"
) -> int:

    if not spwmap:
        spwmap = [-1]

    error = 0

    # Rough bandpass calibration
    caltables_orig = deepcopy(caltables)

    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type="B",
        combine=combine,
        table=f"{msname}_{sourcename}_bcal",
        refant=refant,
        apmode="ap",
        t="inf",
        solnorm=True,
    )
    error += cb.solve()
    error += cb.close()

    caltables += [
        {
            "table": f"{msname}_{sourcename}_bcal",
            "type": "B",
            "spwmap": spwmap,
        }
    ]

    # Gain calibration
    cb = cc.calibrater()
    error += cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type="G",
        combine=combine,
        table=f"{msname}_{sourcename}_gacal",
        refant=refant,
        apmode="a",
        t="inf",
    )
    error += cb.solve()
    error += cb.close()

    caltables += [
        {
            "table": f"{msname}_{sourcename}_gacal",
            "type": "G",
            "spwmap": spwmap,
        }
    ]

    cb = cc.calibrater()
    error += cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type="G",
        combine=combine,
        table=f"{msname}_{sourcename}_gpcal",
        refant=refant,
        apmode="p",
        t="inf",
    )
    error += cb.solve()
    error += cb.close()

    # Final bandpass calibration
    caltables = caltables_orig + [
        {
            "table": f"{msname}_{sourcename}_gacal",
            "type": "G",
            "spwmap": spwmap,
        },
        {
            "table": f"{msname}_{sourcename}_gpcal",
            "type": "G",
            "spwmap": spwmap,
        },
    ]

    cb = cc.calibrater()
    error += cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type="B",
        combine=combine,
        table=f"{msname}_{sourcename}_bacal",
        refant=refant,
        apmode="a",
        t="inf",
        solnorm=True,
    )
    error += cb.solve()
    error += cb.close()

    if not forsystemhealth:
        interpolate_bandpass_solutions(
            msname,
            sourcename,
            mode="a",
        )

    caltables += [
        {
            "table": f"{msname}_{sourcename}_bacal",
            "type": "B",
            "spwmap": spwmap,
        }
    ]

    cb = cc.calibrater()
    error += cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type="B",
        combine=combine,
        table=f"{msname}_{sourcename}_bpcal",
        refant=refant,
        apmode="p",
        t="inf",
        solnorm=True,
    )
    error += cb.solve()
    error += cb.close()

    if not forsystemhealth:
        interpolate_bandpass_solutions(
            msname,
            sourcename,
            mode="p",
        )

    if forsystemhealth:
        caltables += [
            {
                "table": f"{msname}_{sourcename}_bpcal",
                "type": "B",
                "spwmap": spwmap,
            }
        ]
        cb = cc.calibrater()
        error += not cb.open(f"{msname}.ms")
        error += apply_calibration_tables(cb, caltables)
        error += not cb.setsolve(
            type="M" if blbased else "G",
            combine=combine,
            table=f"{msname}_{sourcename}_2gcal",
            refant=refant,
            apmode="ap",
            t=tbeam,
        )
        error += not cb.solve()
        error += not cb.close()

    return error


def calc_delays(vis: np.ndarray, df: float, nfavg: int = 5, tavg: bool = True) -> tuple:
    """Calculates power as a function of delay from the visibilities.

    This uses scipy fftpack to fourier transform the visibilities along the
    frequency axis.  The power as a function of delay can then be used in
    fringe-fitting.

    Parameters
    ----------
    vis : ndarray
        The complex visibilities. 4 dimensions, (baseline, time, frequency,
        polarization).
    df : float
        The width of the frequency channels in GHz.
    nfavg : int
        The number of frequency channels to average by after the Fourier
        transform.  Defaults to 5.
    tavg : boolean
        If ``True``, the visibilities are averaged in time before the Fourier
        transform. Defaults to ``True``.

    Returns
    -------
    vis_ft : ndarray
        The complex visibilities, Fourier-transformed along the time axis. 3
        (or 4, if `tavg` is set to False) dimensions, (baseline, delay,
        polarization) (or (baseline, time, delay, polarization) if `tavg` is
        set to False).
    delay_arr : ndarray
        Float, the values of the delay pixels in nanoseconds.
    """
    nfbins = vis.shape[-2] // nfavg * nfavg
    npol = vis.shape[-1]
    if tavg:
        vis_ft = fftshift(
            fft(
                np.pad(vis[..., :nfbins, :].mean(1), ((0, 0), (0, nfbins), (0, 0))),
                axis=-2,
            ),
            axes=-2,
        )
        vis_ft = vis_ft.reshape(vis_ft.shape[0], -1, 2 * nfavg, npol).mean(-2)
    else:
        vis_ft = fftshift(
            fft(
                np.pad(vis[..., :nfbins, :], ((0, 0), (0, 0), (0, nfbins), (0, 0))),
                axis=-2,
            ),
            axes=-2,
        )
        vis_ft = vis_ft.reshape(
            vis_ft.shape[0], vis_ft.shape[1], -1, 2 * nfavg, npol
        ).mean(-2)
    delay_arr = fftshift(fftfreq(nfbins)) / df
    delay_arr = delay_arr.reshape(-1, nfavg).mean(-1)

    return vis_ft, delay_arr


def apply_calibration(
        msname: str, calname: str, msnamecal: str = None, combine_spw: str = False, nspw: int = 1,
        blbased: bool = False
) -> int:
    r"""Applies the calibration solution.

    Applies delay, bandpass and complex gain tables to a measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set to apply calibration solutions to.
        Opens `msname`.ms
    calname : str
        The name of the calibrator. Tables that start with
        `msnamecal`\_`calname` will be applied to the measurement set.
    msnamecal : str
        The name of the measurement set used to model the calibration solutions
        Calibration tables prefixed with `msnamecal`\_`calname` will be opened
        and applied. If ``None``, `msnamecal` is set to `msname`. Defaults to
        ``None``.
    combine_spw : bool
        Set to True if multi-spw ms and spws in the ms were combined before
        calibration.
    nspw : int
        The number of spws in the ms.
    blbased : bool
        Set to True if the calibration was baseline-based.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combine_spw:
        spwmap = [0] * nspw
    else:
        spwmap = [-1]
    if msnamecal is None:
        msnamecal = msname
    caltables = [
        {
            "table": f"{msnamecal}_{calname}_kcal",
            "type": "K",
            "spwmap": spwmap,
        },
        {
            "table": f"{msnamecal}_{calname}_bcal",
            "type": "MF" if blbased else "B",
            "spwmap": spwmap,
        },
        {
            "table": f"{msname}_{calname}_gacal",
            "type": "M" if blbased else "G",
            "spwmap": spwmap,
        },
        {
            "table": f"{msnamecal}_{calname}_gpcal",
            "type": "M" if blbased else "G",
            "spwmap": spwmap,
        },
    ]
    error = apply_calibration_tables(msname, caltables)
    return error


def apply_delay_bp_cal(
        msname: str, calname: str, blbased: bool = False, msnamecal: str = None,
        combine_spw: bool = False, nspw: int = 1
) -> int:
    r"""Applies delay and bandpass calibration.

    Parameters
    ----------
    msname : str
        The name of the measurement set containing the visibilities. The
        measurement set `msname`.ms will be opened.
    calname : str
        The name of the calibrator source used in calibration of the
        measurement set. The tables `msname`\_`calname`_kcal and
        `msnamecal`\_`calname`_bcal will be applied to the measurement set.
    blbased : boolean
        Set to True if baseline-based calibration routines were done. Defaults
        False.
    msnamecal : str
        The prefix of the measurement set used to derive the calibration
        solutions. If None, set to `msname`.
    combine_spw : boolean
        Set to True if the spws were combined when deriving the solutions.
        Defaults False.
    nspw : int
        The number of spws in the dataset.  Only used if `combine_spw` is set
        to True. Defaults 1.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combine_spw:
        spwmap = [0] * nspw
    else:
        spwmap = [-1]
    if msnamecal is None:
        msnamecal = msname
    error = 0
    caltables = [
        {
            "table": f"{msname}_{calname}_kcal",
            "type": "K",
            "spwmap": spwmap,
        },
        {
            "table": f"{msnamecal}_{calname}_bcal",
            "type": "MF" if blbased else "B",
            "spwmap": spwmap,
        },
    ]
    error += apply_and_correct_calibrations(msname, caltables)
    return error


def calibrate_gain(
        msname: str, calname: str, caltable_prefix: str, refant: str, tga: str, tgp: str,
        blbased: bool = False, combined: bool = False
) -> int:
    """Calculates gain calibration only.

    Uses existing solutions for the delay and bandpass.

    Parameters
    ----------
    msname : str
        The name of the measurement set for gain calibration.
    calname : str
        The name of the calibrator used in calibration.
    caltable_prefix : str
        The prefix of the delay and bandpass tables to be applied.
    refant : str
        The name of the reference antenna.
    tga : str
        A casa-understood integration time for gain amplitude calibration.
    tgp : str
        A casa-understood integration time for gain phase calibration.
    blbased : boolean
        Set to True if using baseline-based calibration for gains. Defaults
        False.
    combined : boolean
        Set to True if spectral windows are combined for calibration.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combined:
        spwmap = [0]
    else:
        spwmap = [-1]
    if blbased:
        gtype = "M"
        bptype = "MF"
    else:
        gtype = "G"
        bptype = "B"
    combine = "scan,field,obs"
    caltables = [
        {"table": f"{caltable_prefix}_kcal", "type": "K", "spwmap": spwmap}
    ]
    error = 0
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type=bptype,
        combine=combine,
        table=f"{msname}_{calname}_bcal",
        minblperant=1,
        refant=refant,
    )
    error += not cb.solve()
    error += not cb.close()
    caltables += [
        {"table": f"{caltable_prefix}_bcal", "type": bptype, "spwmap": spwmap}
    ]
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type=gtype,
        combine=combine,
        table=f"{msname}_{calname}_gpcal",
        t=tgp,
        minblperant=1,
        refant=refant,
        apmode="p",
    )
    error += not cb.solve()
    error += not cb.close()
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    caltables += [
        {
            "table": f"{msname}_{calname}_gpcal",
            "type": gtype,
            "spwmap": spwmap,
        }
    ]
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type=gtype,
        combine=combine,
        table=f"{msname}_{calname}_gacal",
        t=tga,
        minblperant=1,
        refant=refant,
        apmode="a",
    )
    error += not cb.solve()
    error += not cb.close()
    return error


def apply_and_correct_calibrations(msname: str, calibration_tables: List[dict]) -> int:
    """Applies and corrects calibration tables in an ms.

    Parameters
    ----------
    msname : str
        The measurement set filepath. Will open `msname`.ms.
    calibration_tables : list
        Calibration tables to apply. Each entry is a dictionary containing the
        keywords 'type' (calibration type, e.g. 'K'), 'spwmap' (spwmap for the
        calibration), and 'table' (full path to the calibration table).

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    error = 0
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += apply_calibration_tables(cb, calibration_tables)
    error += not cb.correct()
    error += not cb.close()
    return error


def apply_calibration_tables(cb: "cc.calibrater", calibration_tables: List[dict]) -> int:
    """Applies calibration tables to an open calibrater object.

    Parameters
    ----------
    cb : cc.calibrater() instance
        Measurement set should be opened already.
    calibration_tables : list
        Calibration tables to apply. Each entry is a dictionary containing the
        keywords 'type' (calibration type, e.g. 'K'), 'spwmap' (spwmap for the
        calibration), and 'table' (full path to the calibration table).

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    error = 0
    for caltable in calibration_tables:
        error += not cb.setapply(
            type=caltable["type"], spwmap=caltable["spwmap"], table=caltable["table"]
        )
    return error


def interpolate_bandpass_solutions(
        msname: str, calname: str, thresh: float = 1.5, polyorder: int = 7, mode: str = "ap"
) -> None:
    r"""Interpolates bandpass solutions.

    Parameters
    ----------
    msname : str
        The measurement set filepath (with the `.ms` extension omitted).
    calname : str
        The name of the calibrator source. Calibration tables starting with
        `msname`\_`calname` will be opened.
    thresh : float
        Sets flagging of bandpass solutions before interpolating in order to
        smooth the solutions. After median baselining, any points that deviate
        by more than interp_thresh*std are flagged.
    polyorder : int
        The order of the polynomial used to smooth bandpass solutions.
    mode : str
        The bandpass calibration mode. Must be one of "a", "p" or "ap".
    """
    if mode == "a":
        tbname = "bacal"
    elif mode == "p":
        tbname = "bpcal"
    elif mode == "ap":
        tbname = "bcal"
    else:
        raise RuntimeError('mode must be one of "a", "p" or "ap"')

    with table(f"{msname}_{calname}_{tbname}") as tb:
        bpass = np.array(tb.CPARAM[:])
        flags = np.array(tb.FLAG[:])

    with table(f"{msname}.ms") as tb:
        antennas = np.unique(np.array(tb.ANTENNA1[:]))

    with table(f"{msname}.ms/SPECTRAL_WINDOW") as tb:
        fobs = np.array(tb.CHAN_FREQ[:]).squeeze(0) / 1e9

    bpass_amp = np.abs(bpass)
    bpass_ang = np.angle(bpass)
    bpass_amp_out = np.ones(bpass.shape, dtype=bpass.dtype)
    bpass_ang_out = np.zeros(bpass.shape, dtype=bpass.dtype)

    # Interpolate amplitudes
    if mode in ("a", "ap"):
        std = bpass_amp.std(axis=1, keepdims=True)
        for ant in antennas:
            for j in range(bpass.shape[-1]):
                offset = (
                    np.abs(
                        bpass_amp[ant - 1, :, j] - medfilt(bpass_amp[ant - 1, :, j], 9)
                    )
                    / std[ant - 1, :, j]
                )
                idx = offset < thresh
                idx[flags[ant - 1, :, j]] = 1
                if sum(idx) > 0:
                    z_fit = np.polyfit(fobs[idx], bpass_amp[ant - 1, idx, j], polyorder)
                    p_fit = np.poly1d(z_fit)
                    bpass_amp_out[ant - 1, :, j] = p_fit(fobs)

    # Interpolate phase
    if mode in ("p", "ap"):
        std = bpass_ang.std(axis=1, keepdims=True)
        for ant in antennas:
            for j in range(bpass.shape[-1]):
                offset = (
                    np.abs(
                        bpass_ang[ant - 1, :, j] - medfilt(bpass_ang[ant - 1, :, j], 9)
                    )
                    / std[ant - 1, :, j]
                )
                idx = offset < thresh
                idx[flags[ant - 1, :, j]] = 1
                if sum(idx) > 0:
                    z_fit = np.polyfit(fobs[idx], bpass_ang[ant - 1, idx, j], 7)
                    p_fit = np.poly1d(z_fit)
                    bpass_ang_out[ant - 1, :, j] = p_fit(fobs)

    with table(f"{msname}_{calname}_{tbname}", readonly=False) as tb:
        tb.putcol("CPARAM", bpass_amp_out * np.exp(1j * bpass_ang_out))
        # Reset flags for the interpolated solutions
        tbflag = np.array(tb.FLAG[:])
        tb.putcol("FLAG", np.zeros(tbflag.shape, tbflag.dtype))


def calibrate_phases(
        filenames: dict, refant: str, msdir: str = "/mnt/data/dsa110/calibration/"
) -> None:
    """Calibrate phases only for a group of calibrator passes.

    Parameters
    ----------
    filenames : dict
        A dictionary containing information on the calibrator passes to be
        calibrated. Same format as dictionary returned by
        dsacalib.utils.get_filenames()
    refant : str
        The reference antenna name to use. If int, will be interpreted as the
        reference antenna index instead.
    msdir : str
        The full path to the measurement set, with the `.ms` extension omitted.
    """
    for date in filenames.keys():
        for cal in filenames[date].keys():
            msname = f"{msdir}/{date}_{cal}"
            if os.path.exists(f"{msname}.ms"):
                cb = cc.calibrater()
                cb.open(f"{msname}.ms")
                cb.setsolve(
                    type="B",
                    combine="field,scan,obs",
                    table=f"{msname}_{cal}_bpcal",
                    refant=refant,
                    apmode="p",
                    t="inf",
                )
                cb.solve()
                cb.close()


def calibrate_phase_single_ms(msname: str, refant: str, calname: str) -> None:
    """Generate a bandpass gain calibration table for a single ms.

    Parameters
    ----------
    msname : str
        Will open `msname`.ms
    refant : str
        The reference antenna name.
    calname : str
        The calibrator name.
    """
    cb = cc.calibrater()
    cb.open(f"{msname}.ms")
    cb.setsolve(
        type="B",
        combine="field,scan,obs",
        table=f"{msname}_{calname}_bpcal",
        refant=refant,
        apmode="p",
        t="inf",
    )
    cb.solve()
    cb.close()


def combine_bandpass_and_delay(table_prefix: str, forsystemhealth: bool) -> None:
    """Combine bandpass and delay tables into a single bandpass table."""
    with table(f"{table_prefix}_bacal") as tb:
        bpass = np.array(tb.CPARAM[:])
    with table(f"{table_prefix}_bpcal") as tb:
        bpass *= np.array(tb.CPARAM[:])

    if not forsystemhealth:
        with table(f"{table_prefix}_bkcal") as tb:
            bpass *= np.array(tb.CPARAM[:])

    if not os.path.exists(f"{table_prefix}_bcal"):
        tablecopy(f"{table_prefix}_bpcal", f"{table_prefix}_bcal")

    with table(f"{table_prefix}_bcal", readonly=False) as tb:
        tb.putcol("CPARAM", bpass)
        if not forsystemhealth:
            tbflag = np.array(tb.FLAG[:])
            tb.putcol("FLAG", np.zeros(tbflag.shape, tbflag.dtype))
