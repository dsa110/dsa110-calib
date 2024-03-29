"""Monitor the primary beam pointing and send voltage triggers on continuum sources.

Aim for ~3 calibrator sources a day (weighted flux > 0.2 of the total field).
One trigger on each source.
Stick close to the centre of the primary beam (+/- 2.5 deg).
"""

import datetime
import signal
from functools import wraps
from typing import Callable
from threading import Event

import astropy.units as u
from astropy.time import Time
import numpy as np
import pandas

import dsautils.dsa_store as ds
from dsacalib.preprocess import update_caltable

SECONDS_PER_SIDEREAL_DAY = 86164.0905*u.s/(360*u.deg)

EXIT_EVENT = Event()

def continuum_voltage_triggers():
    """Send a voltage trigger when continuum sources is in primary beam.

    Triggers on 3 sources with the most weighted flux (as percent of the field)
    at each pointing declination.
    """
    etcd = ds.DsaStore()

    # Initialize calsources
    calsources = None

    def update_caltable_callback(etcd_dict: dict) -> None:
        """When the antennas are moved, and and read a new calibration table.

        watch /mon/array/dec
        """
        nonlocal calsources
        caltable = update_caltable(etcd_dict['dec_deg']*u.deg)
        calsources = pandas.read_csv(caltable, header=0)
        calsources.sort_values("percent flux", inplace=True, ascending=False)
        assert len(calsources) > 0, "No continuum sources at current dec"
        calsources = calsources.head(3)
        calsources.reset_index(inplace=True, drop=True)

        print(f"Updated calsources to\n {calsources}")

    update_caltable_callback(etcd.get_dict('/mon/array/dec'))
    etcd.add_watch('/mon/array/dec', update_caltable_callback)

    check_sources_and_trigger_loop = run_and_wait(check_sources_and_trigger, 5*60)
    check_sources_and_trigger_loop(calsources, etcd)


def check_sources_and_trigger(calsources: pandas.DataFrame, etcd: "etcd object") -> None:
    """Determines if any sources are in the primary beam, and sends voltage triggers if any are."""
    current_pointing = etcd.get_dict('/mon/array/pointing_J2000')
    time_to_transit = (
        calsources['ra'].to_numpy()*u.deg - current_pointing['ra_deg']*u.deg
    )*SECONDS_PER_SIDEREAL_DAY

    to_trigger = np.where((time_to_transit > -2.5*u.min) & (time_to_transit < 2.5*u.min))[0]

    for idx in to_trigger:
        print(f"Triggering on {calsources.loc[idx, 'source']}")
        etcd.put_dict(
            '/cmd/corr/0',
            {
                'cmd': 'ctrltrigger',
                'val': Time.now().strftime('%y%m%d') + calsources.loc[idx, 'source']
            })


def run_and_wait(target: Callable, frequency_s: int) -> Callable:
    """Returns Callable that executes the target, but only returns once frequency_s has elapsed."""

    @wraps(target)
    def inner(*args, **kwargs):
        while not EXIT_EVENT.is_set():
            start = datetime.datetime.utcnow()

            print(f"Running {target.__name__}")
            target(*args, **kwargs)

            elapsed = (datetime.datetime.utcnow() - start).total_seconds()
            tosleep = frequency_s - elapsed
            if tosleep > 0:
                EXIT_EVENT.wait(tosleep)

    return inner


def quit_service(signo, _frame):
    """Quit the service."""
    print(f"Interrupted by {signo}, shutting down")
    EXIT_EVENT.set()


if __name__ == '__main__':

    for sig in ('TERM', 'HUP', 'INT'):
        signal.signal(getattr(signal, 'SIG'+sig), quit_service)

    continuum_voltage_triggers()
