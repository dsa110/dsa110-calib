"""Dedisperse a measurement set.
"""

import astropy.units as u
import numpy as np
import scipy  # pylint: disable=unused-import # must come before casacore
from casacore.tables import table
from numba import jit

from dsacalib.ms_io import extract_vis_from_ms

# Same as tempo2
# TODO: Verify same DM constant used in T1
DISPERSION_DELAY_CONSTANT = u.s / 2.41e-4 * u.MHz**2 * u.cm**3 / u.pc


def time_delay(dispersion_measure, freq, ref_freq):
    r"""Time delay due to dispersion.

    Parameters
    ----------
    dispersion_measure : astropy quantity
        Dispersion measure in units of pc / cm3.
    freq : astropy quantity
        Frequency at which to evaluate the dispersion delay.
    ref_freq : astropy quantity
        Reference frequency relative to which the dispersion delay is
        evaluated.

    Returns
    -------
    quantity
        The dispersion delay of `freq` relative to `ref_freq`.
    """
    ref_freq_inv2 = 1 / ref_freq**2
    return (
        dispersion_measure * DISPERSION_DELAY_CONSTANT * (1 / freq**2 - ref_freq_inv2)
    ).to(u.ms)


def disperse(msname, dispersion_measure, ref_freq=1.405 * u.GHz):
    """Disperses measurement set incoherently.

    Parameters
    ----------
    msname : str
        The full path to the measurement set to dedisperse, without the `.ms`
        extension.
    dispersion_measure : astropy quantity
        Dispersion measure to remove in pc / cm3.
    ref_freq : astropy quantity
        Reference frequency relative to which the dispersion delay is
        evaluated.
    """
    # TODO: Pad to reduce corruption of signal at the edges.
    # Get data from ms
    data, time, freq, _, _, _, _, spw, orig_shape = extract_vis_from_ms(
        msname, swapaxes=False
    )
    spwaxis = orig_shape.index("spw")
    timeaxis = orig_shape.index("time")
    # To speed up dedispersion, we are hardcoding for the order of the axes
    # Check that the order is consistent with our assumptions
    assert spwaxis == 2
    assert timeaxis == 0
    freq = freq.reshape(-1, data.shape[3]) * u.GHz
    freq = freq[spw, :]
    time = ((time - time[0]) * u.d).to(u.ms)
    dtime = np.diff(time)
    assert np.all(np.abs((dtime - dtime[0]) / dtime[0]) < 1e-2)
    dtime = np.median(dtime)

    # Calculate dispersion delay and roll axis
    dispersion_delay = time_delay(dispersion_measure, freq, ref_freq)
    dispersion_bins = np.rint(dispersion_delay / dtime).astype(np.int)
    # Jit provides a very moderate speed up of ~6 percent
    disperse_worker(data, dispersion_bins)

    # Write out the data to the ms
    data = data.reshape(-1, data.shape[3], data.shape[4])
    with table(f"{msname}.ms", readonly=False) as tb:
        tb.putcol("DATA", data)


def dedisperse(msname, dispersion_measure, ref_freq=1.405 * u.GHz):
    """Dedisperses measurement set incoherently.

    Parameters
    ----------
    msname : str
        The full path to the measurement set to dedisperse, without the `.ms`
        extension.
    dispersion_measure : astropy quantity
        Dispersion measure to remove in pc / cm3.
    ref_freq : astropy quantity
        Reference frequency relative to which the dispersion delay is
        evaluated.
    """
    disperse(msname, -1 * dispersion_measure, ref_freq)


@jit(nopython=True)
def disperse_worker(data, dispersion_bins):
    """Roll with jit.

    Parameters
    ----------
    data : ndarray
        Dimensions (time, baseline, spw, freq, pol).
    dispersion_bins : ndarray
        The number of bins to shift by. Dimensions (spw, freq).
    """
    for bidx in range(data.shape[1]):
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                for pidx in range(data.shape[4]):
                    data[:, bidx, i, j, pidx] = np.roll(
                        data[:, bidx, i, j, pidx],
                        dispersion_bins[i, j],
                    )


def disperse_python(data, dispersion_bins):
    """Numpy roll.

    Parameters
    ----------
    data : ndarray
        Dimensions (time, baseline, spw, freq, pol).
    dispersion_bins : ndarray
        The number of bins to shift by. Dimensions (spw, freq).
    """
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            data[..., i, j, :] = np.roll(
                data[..., i, j, :], dispersion_bins[i, j], axis=0
            )
