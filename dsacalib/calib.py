"""Functions for calibration of DSA-110 visibilities.

These functions use the CASA package casatools to calibrate
visibilities stored as measurement sets.

Author: Dana Simard, dana.simard@astro.caltech.edu, 10/2019

"""
import os
import shutil
from typing import List, Tuple
from copy import deepcopy

# Always import scipy before casatools
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import savgol_filter

import numpy as np
import casatools as cc
from casacore.tables import table, tablecopy

from dsacalib.ms_io import read_caltable, freq_GHz_from_ms


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
    error += not cb.selectvis()
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
        msname: str, sourcename: str, refant: str, caltables: List[dict],
        combine: str = "field,scan,obs", spwmap: List = None,
        tbeam: str = "60s"
) -> int:
    """Solve for gain calibration after applying `caltables`.

    This includes:
    Rough bandpass calibration (bcal)
    Gain calibration (gacal, then gpcal)
    Bandpass calibration (bacal, then bpcal)
    Gain calibration on short timescales of tbeam (2gcal)
    """
    if not spwmap:
        spwmap = [-1]

    error = 0

    caltables_orig = deepcopy(caltables)

    # Gain calibration - amplitude
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += not cb.selectvis()
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type="G", combine=combine, table=f"{msname}_{sourcename}_gacal", refant=refant,
        apmode="a", t="inf")
    error += not cb.solve()
    error += not cb.close()

    # Gain calibration - phase
    caltables += [
        {
            "table": f"{msname}_{sourcename}_gacal",
            "type": "G",
            "spwmap": spwmap}]
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += not cb.selectvis()
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type="G", combine=combine, table=f"{msname}_{sourcename}_gpcal", refant=refant, apmode="p",
        t="inf")
    error += not cb.solve()
    error += not cb.close()

    # Gain calibration - short timescales
    caltables = caltables_orig
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += not cb.selectvis()
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type="G", combine=combine, table=f"{msname}_{sourcename}_2gcal", refant=refant,
        apmode="ap", t=tbeam)
    error += not cb.solve()
    error += not cb.close()

    return error

def bandpass_calibration(
        msname: str, sourcename: str, refant: str, caltables: List[dict],
        combine: str = "field,scan,obs", spwmap: List = None) -> int:
    """Calibrate the bandpass.

    Parameters
    ----------
    msname : str
        The path to the measurement set, omitting the final ".ms"
    sourcename : str
        The name of the calibrator, at which the ms is phased.
    refant : str
        The reference antenna name to use.
    caltables : List[dict]
        The tables to be applied prior to bandpass calibration.  Should include
        "table", "type" and "spwmap" keys.
    combine : str
        The casa combine parameter for solving calibration.
    spwmap : List[int]
        The casa spwmap parameter. e.g. [-1] beams apply each spw's solution to the same
        spw in the ms.  [0]*nspws means apply spw 0's solution to every spw.

    Returns
    -------
    int : the number of errors that occured in the casa calls
    """
    if not spwmap:
        spwmap = [-1]

    error = 0

    # Amplitude
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += not cb.selectvis()
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type="B", combine=combine, table=f"{msname}_{sourcename}_bacal",
        refant=refant, apmode="a", t="inf", solnorm=True)
    error += not cb.solve()
    error += not cb.close()

    # Phase
    caltables += [
        {
            "table": f"{msname}_{sourcename}_bacal",
            "type": "B",
            "spwmap": spwmap}]
    cb = cc.calibrater()
    error += not cb.open(f"{msname}.ms")
    error += not cb.selectvis()
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(
        type="B", combine=combine, table=f"{msname}_{sourcename}_bpcal",
        refant=refant, apmode="p", t="inf", solnorm=True)
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
    error += not cb.selectvis()
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
    error += not cb.selectvis()
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
    error += not cb.selectvis()
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
                cb.selectvis()
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


def calculate_bandpass_from_all_tables(
        msname: str, table_prefix: str, delay_bandpass_table_prefix: str = "",
        filter_phase: bool = True) -> Tuple[np.ndarray]:
    """Combines gain, bandpass, and delay tables into a single bandpass.

    If `filter_phase` is set to `True`, then use savgol filter to smooth the bandpass phases,
    and channel flags are not propagated to the returned flags array.
    """
    if not delay_bandpass_table_prefix:
        delay_bandpass_table_prefix = table_prefix
    fobs = freq_GHz_from_ms(msname)
    fmean = np.mean(fobs)

    kcal, _, kflags, *_ = read_caltable(f"{delay_bandpass_table_prefix}_kcal", reshape=False)
    bacal, _, baflags, *_ = read_caltable(
        f"{delay_bandpass_table_prefix}_bacal", reshape=False, cparam=True)
    bpcal, _, bpflags, *_ = read_caltable(
        f"{delay_bandpass_table_prefix}_bpcal", reshape=False, cparam=True)
    gacal, _, gaflags, *_ = read_caltable(f"{table_prefix}_gacal", reshape=False, cparam=True)
    gpcal, _, gpflags, *_ = read_caltable(f"{table_prefix}_gpcal", reshape=False, cparam=True)

    if filter_phase:
        bpcal = (
            np.ones(bpcal.shape, bpcal.dtype) *
            np.exp(1j * savgol_filter(np.angle(bpcal), 11, 1, axis=1)))

    flags = kflags | gaflags | gpflags
    flags = np.tile(flags, (1, bpcal.shape[1], 1))
    if filter_phase:
        flags = flags | baflags | bpflags

    bandpass = (
        bacal * bpcal * gacal * gpcal *
        np.exp(2j*np.pi * (fobs[:, np.newaxis] - fmean) * kcal))

    return bandpass, flags


def combine_tables(
        msname: str, table_prefix: str, delay_bandpass_table_prefix: str = "",
        filter_phase: bool = True) -> None:
    """Combine gain, bandpass and delay tables into a single bandpass table."""

    bandpass, flags = calculate_bandpass_from_all_tables(
        msname, table_prefix, delay_bandpass_table_prefix, filter_phase)

    if not os.path.exists(f"{table_prefix}_bcal"):
        tablecopy(f"{table_prefix}_bpcal", f"{table_prefix}_bcal")

    with table(f"{table_prefix}_bcal", readonly=False) as tb:
        tb.putcol("CPARAM", bandpass)
        tb.putcol("FLAG", flags)
