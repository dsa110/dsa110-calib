import astropy.units as u

import dsautils.dsa_store as ds
from dsacalib.preprocess import update_caltable


# Aim for ~3 calibrator sources a day
# One trigger on each source 
# Stick close to the centre of the primary beam

# TODO: decide on flux scale, check how to put a reasonable trigger name

def continuum_voltage_triggers():
    seconds_per_sidereal_day = 86164.0905*u.s/(360*u.deg)
    etcd = ds.DsaStore()

    # Initialize calsources
    calsources = None

    def update_caltable_callback(etcd_dict: dict) -> None:
        """When the antennas are moved, and and read a new calibration table.

        watch /mon/array/dec
        """
        nonlocal calsources
        caltable = update_caltable(etcd_dict['pt_dec_deg']*u.deg)
        calsources = pandas.read_csv(caltable, header=0)

    update_caltable_callback(etcd.get_dict('/mon/array/dec'))
    etcd.add_watch('/mon/array/dec', update_caltable_callback)

    while True:
        current_pointing = etcd.get_dict('/mon/array/pointing_J2000')
        time_to_transit = (calsources['ra']*u.deg - current_pointing['ra_deg']*u.deg)*seconds_per_sidereal_day
        # Check times and send or queue voltage triggers dumps without about 10 s
        if time_to_transit > -2.5*u.min and time_to_transit < 2.5*u.min:
            etcd.put_dict('/cmd/corr/0', {'cmd': 'ctrltrigger', 'val': calsources['name']})

        time.sleep(5*60)


if __name__ == '__main__':
    continuum_voltage_triggers()
