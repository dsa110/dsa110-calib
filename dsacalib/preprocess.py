"""Functions used in preprocessing of hdf5 files.

Prepares hdf5 files written by dsa-meridian-fs for conversion to ms.
"""
import os
import re
import subprocess
from urllib.request import urlretrieve
from collections import namedtuple

import astropy.units as u
from astropy.coordinates import Angle
import numpy as np
import pandas
from pkg_resources import resource_exists, resource_filename
from pyuvdata import UVData

from dsacalib.fringestopping import pb_resp


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns ``default``
    If ``pred`` is not None, returns the first item
    for which pred(item) is true.

    Parameters
    ----------
    iterable : list
        The list for which to find the first True item.
    default :
        If not False, then this is returned if no true value is found.
        Defaults False.
    pred :
        If not None, then the first item for which pred(item) is True is
        returned.
    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


def rsync_file(rsync_string, remove_source_files=True, logger=None):
    """Rsyncs a file from the correlator machines to dsastorage.

    Parameters
    ----------
    rsync_string : str
        E.g. 'corr06.sas.pvt:/home/ubuntu/data/2020-06-24T12:32:06.hdf5 '
        '/mnt/data/dsa110/correlator/corr06/'
    """
    fname, fdir = rsync_string.split(" ")
    if remove_source_files:
        command = (
            f". ~/.keychain/calibration-sh; rsync -avv --remove-source-files "
            f"{fname} {fdir}")
    else:
        command = f". ~/.keychain/calibration-sh; rsync -avv {fname} {fdir}"
    with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True
    ) as process:
        proc_stdout = str(process.communicate()[0].strip())

    if logger:
        logger.info(proc_stdout)

    fname = fname.split("/")[-1]
    return f"{fdir}{fname}"


def remove_outrigger_delays(UVhandler, outrigger_delays):
    """Remove outrigger delays from open UV object."""

    if "applied_delays_ns" in UVhandler.extra_keywords.keys():
        applied_delays = (
            np.array(UVhandler.extra_keywords["applied_delays_ns"].split(" "))
            .astype(np.int)
            .reshape(-1, 2)
        )
    else:
        applied_delays = np.zeros((UVhandler.Nants_telescope, 2), np.int)
    fobs = UVhandler.freq_array * u.Hz
    # Remove delays for the outrigger antennas
    for ant, delay in outrigger_delays.items():
        phase_model = np.exp(
            (2.0j * np.pi * fobs * (delay * u.nanosecond)).to_value(
                u.dimensionless_unscaled
            )
        ).reshape(1, fobs.shape[0], fobs.shape[1], 1)
        UVhandler.data_array[
            (UVhandler.ant_1_array != UVhandler.ant_2_array)
            & (UVhandler.ant_2_array == ant - 1)
        ] *= phase_model
        UVhandler.data_array[
            (UVhandler.ant_1_array != UVhandler.ant_2_array)
            & (UVhandler.ant_1_array == ant - 1)
        ] /= phase_model
        applied_delays[ant - 1, :] += int(delay)
    if "applied_delays_ns" in UVhandler.extra_keywords.keys():
        UVhandler.extra_keywords["applied_delays_ns"] = np.string_(
            " ".join([str(d) for d in applied_delays.flatten()])
        )


def fscrunch_file(fname, nfreq_scrunch, outrigger_delays):
    """Removes outrigger delays before averaging in frequency.

    Leaves file untouched if the number of frequency bins is not divisible
    by the desired number of frequency bins (nfreq), or is equal to the desired
    number of frequency bins.

    Parameters
    ----------
    fname : str
        The full path to the file to process.
    """
    # Process the file
    # print(fname)
    UV = UVData()
    UV.read_uvh5(fname, run_check_acceptability=False)
    nint = UV.Nfreqs // nfreq_scrunch
    if nint > 1 and UV.Nfreqs % nint == 0:
        remove_outrigger_delays(UV, outrigger_delays)
        # Scrunch in frequency by factor of nint
        UV.frequency_average(n_chan_to_avg=nint)
        if os.path.exists(fname.replace(".hdf5", "_favg.hdf5")):
            os.remove(fname.replace(".hdf5", "_favg.hdf5"))
        # Overwrite
        UV.write_uvh5(fname, run_check_acceptability=False)
    return fname


def read_nvss_catalog():
    """
    Reads the NVSS catalog into a pandas dataframe.
    Returns flux in mJy.
    """
    if not resource_exists("dsacalib", "data/heasarc_nvss.tdat"):
        urlretrieve(
            (
                "https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/tdat_files/"
                "heasarc_nvss.tdat.gz"),
            resource_filename("dsacalib", "data/heasarc_nvss.tdat.gz"),
        )
        os.system((
            f"gunzip "
            f"{resource_filename('dsacalib', 'data/heasarc_nvss.tdat.gz')}"))

    df = pandas.read_csv(
        resource_filename("dsacalib", "data/heasarc_nvss.tdat"),
        sep="|",
        skiprows=67,
        names=[
            "ra",
            "dec",
            "lii",
            "bii",
            "ra_error",
            "dec_error",
            "flux_20_cm",
            "flux_20_cm_error",
            "limit_major_axis",
            "major_axis",
            "major_axis_error",
            "limit_minor_axis",
            "minor_axis",
            "minor_axis_error",
            "position_angle",
            "position_angle_error",
            "residual_code",
            "residual_flux",
            "pol_flux",
            "pol_flux_error",
            "pol_angle",
            "pol_angle_error",
            "field_name",
            "x_pixel",
            "y_pixel",
            "extra",
        ],
    )
    df.drop(df.tail(1).index, inplace=True)
    df.drop(["extra"], axis="columns", inplace=True)
    return df


def read_vla_catalog():
    """Read the VLA calibrator list into a dataframe.

    Kept source, ra, dec, flux_20_cm keys in the NVSS catalog
    so we can easily switch the two of them out.
    flux is in mJy.
    """

    Calibrator = namedtuple(
        "Calibrator", "source ra dec flux_20_cm code_20_cm")
    filename = resource_filename("dsacalib", "data/vlacalibrators.txt")
    calsources = []
    with open(filename) as file:
        for _ in range(3):
            file.readline()
        while True:
            line = file.readline()
            if not line:
                break

            source, _, _, ra, dec, *_ = line.split()
            ra = Angle(ra).to_value(u.deg)
            dec = Angle(dec).to_value(u.deg)
            flux_20_cm = None
            code_20_cm = None
            for _ in range(4):
                file.readline()

            while True:
                line = file.readline()
                if line.isspace() or not line:
                    # We've reached the end of an entry
                    if flux_20_cm not in [None, '?']:
                        calsources += [
                            Calibrator(
                                source, ra, dec, 1000 * float(flux_20_cm),
                                code_20_cm)]
                    break
                if "20cm " in line:
                    (
                        _, _, code_a, code_b, code_c, code_d, flux_20_cm, *_
                    ) = line.split()
                    code_20_cm = code_a + code_b + code_c + code_d
    df = pandas.DataFrame.from_records(calsources, columns=Calibrator._fields)
    df.set_index('source', inplace=True)
    return df


def generate_caltable(
        pt_dec, csv_string, radius=2.5 * u.deg, min_weighted_flux=1 * u.Jy,
        min_percent_flux=0.15, codes=None):
    """Generate a table of calibrators at a given declination.

    Parameters
    ----------
    pt_dec : astropy quantity
        The pointing declination, in degrees or radians.
    radius : astropy quantity
        The radius of the DSA primary beam. Only sources out to this radius
        from the pointing declination are considered.
    min_weighted_flux : astropy quantity
        The minimum primary-beam response-weighted flux of a calibrator for
        which it is included in the calibrator list, in Jy or equivalent.
    min_percent_flux : float
        The minimum ratio of the calibrator weighted flux to the weighted flux
        in the primary beam for which to include the calibrator.
    """
    if codes is None:
        codes = ["P", "S"]

    df = read_vla_catalog()
    calibrators = df[
        (df["dec"] < (pt_dec + radius).to_value(u.deg))
        & (df["dec"] > (pt_dec - radius).to_value(u.deg))
        & (df["flux_20_cm"].astype("float") > 1000)
    ]
    # Calculate field flux and weighted flux for each calibrator
    calibrators = calibrators.assign(field_flux=np.zeros(len(calibrators)))
    calibrators = calibrators.assign(weighted_flux=np.zeros(len(calibrators)))
    for name, row in calibrators.iterrows():
        calibrators.loc[name, "weighted_flux"] = (
            row["flux_20_cm"] / 1e3 * pb_resp(
                row["ra"] * (1 * u.deg).to_value(u.rad),
                pt_dec.to_value(u.rad),
                row["ra"] * (1 * u.deg).to_value(u.rad),
                row["dec"] * (1 * u.deg).to_value(u.rad),
                1.4))
        field = df[
            (df["dec"] < (pt_dec + radius).to_value(u.deg))
            & (df["dec"] > (pt_dec - radius).to_value(u.deg))
            & (df["ra"] < row["ra"] + radius.to_value(u.deg) / np.cos(pt_dec))
            & (df["ra"] > row["ra"] - radius.to_value(u.deg) / np.cos(pt_dec))
        ]
        field = field.assign(weighted_flux=np.zeros(len(field)))
        for fname, frow in field.iterrows():
            field.loc[fname, "weighted_flux"] = (
                frow["flux_20_cm"]
                / 1e3
                * pb_resp(
                    row["ra"] * (1 * u.deg).to_value(u.rad),
                    pt_dec.to_value(u.rad),
                    frow["ra"] * (1 * u.deg).to_value(u.rad),
                    frow["dec"] * (1 * u.deg).to_value(u.rad),
                    1.4,
                )
            )
        calibrators.loc[name, "field_flux"] = sum(field["weighted_flux"])

    # Calculate percent of the field flux that is contained in the
    # main calibrator
    calibrators = calibrators.assign(
        percent_flux=calibrators["weighted_flux"] / calibrators["field_flux"]
    )
    # Keep calibrators based on the weighted flux and percent flux
    calibrators = calibrators[
        (calibrators["weighted_flux"] > min_weighted_flux.to_value(u.Jy))
        & (calibrators["percent_flux"] > min_percent_flux)
        & [v[2] in codes for v in calibrators.loc[:, 'code_20_cm']]
        #       & (calibrators["code_20_cm"][2] == code)  # c-config code match
    ]

    # Create the caltable needed by the calibrator service
    caltable = calibrators[[
        "ra", "dec", "flux_20_cm", "weighted_flux", "percent_flux",
        "code_20_cm"]]
    caltable.reset_index(inplace=True)
    caltable.rename(
        columns={
            "index": "source",
            "flux_20_cm": "flux (Jy)",
            "weighted_flux": "weighted flux (Jy)",
            "percent_flux": "percent flux"},
        inplace=True)
    caltable.loc[:, "flux (Jy)"] = caltable["flux (Jy)"] / 1e3
    caltable.loc[:, "source"] = [sname.strip(
        "NVSS ") for sname in caltable["source"]]
    caltable.loc[:, "ra"] = caltable["ra"] * u.deg
    caltable.loc[:, "dec"] = caltable["dec"] * u.deg

    caltable.to_csv(resource_filename("dsacalib", csv_string))


def update_caltable(pt_dec):
    """Updates caltable to new elevation.

    If needed, a new caltable is written to the dsacalib data directory.
    The caltable to be used is copied to 'calibrator_sources.csv' in the
    dsacalib data directory.

    Parameters
    ----------
    pt_el : astropy quantity
        The antenna pointing elevation in degrees or equivalent.
    """
    decsign = "+" if pt_dec.to_value(u.deg) >= 0 else "-"
    decval = f"{np.abs(pt_dec.to_value(u.deg)):05.1f}".replace(".", "p")
    csv_string = f"data/calibrator_sources_dec{decsign}{decval}.csv"
    if not resource_exists("dsacalib", csv_string):
        generate_caltable(pt_dec, csv_string)
    return resource_filename("dsacalib", csv_string)
