from pathlib import Path
from astropy.time import Time
from astropy.units import Quantity
from dsacalib.realtime_calibration import H5File, Scan, ScanCache
from dsacalib.utils import CalibratorSource
import dsacalib

TEST_TIMESTAMP = '2022-07-13T17:15:26'
TEST_PATH = Path(dsacalib.__path__[0])/'data/test/'


def test_H5File():
    filepath = TEST_PATH / f"{TEST_TIMESTAMP}_sb00.hdf5"
    h5file = H5File(filepath)
    assert isinstance(h5file.path, Path)
    assert isinstance(h5file.timestamp, str)
    assert isinstance(h5file.start_time, Time)
    assert h5file.remote_path is None
    assert h5file.hostname is None
    assert isinstance(h5file.pointing_dec, Quantity)


def test_Scan_init():
    h5files = [H5File(TEST_PATH / f"{TEST_TIMESTAMP}_sb{subband:02d}.hdf5") for subband in range(16)]
    scan = Scan(h5files[:8])
    for h5file in h5files[8:]:
        scan.add(h5file)
    assert isinstance(scan.start_time, Time)
    assert scan.nfiles == 16
    assert scan.source is None
    assert all(file is not None for file in scan.files)


def test_Scan_check_for_source():
    pass


def test_Scan_convert_to_ms(tmpdir):
    config = Configuration()
    refmjd = config.refmjd
    
    trigname = 'test'
    caltime = Time(TEST_TIMESTAMP)
    callst = caltime.sidereal_time('apparent', longitude=ct.OVRO_LON)
    hdf5dir = Path(dsacalib.__path__[0])/"data/test"
    h5files = [H5File(h5f) for h5f in sorted(list(hdf5dir.glob(f"{TEST_TIMESTAMP}*.hdf5")))]
    calsource = {
        'source': trigname,
        'ra': callst.to_value(u.deg),
        'dec': first_true(h5files).pointing_dec.to_value(u.deg)
    }
    scan = Scan(h5files, calsource)
    
    msname, cal = scan.convert_to_ms(hdf5dir, refmjd)
    print(msname, cal)
    assert (Path(tmpdir)/f"{caltime.strftime('%Y-%m-%d')}_test.ms").exists()
    assert Path(msname) == Path(tmpdir)/f"{caltime.strftime('%Y-%m-%d')}_test"
    assert isinstance(cal, CalibratorSource)
    assert cal.name == 'test'


def test_ScanCache():
    scancache = ScanCache(12)
    assert scancache.max_scans == 12
    assert all(scan is None for scan in scancache.scans)
    assert len(scancache.scans) == 12
    assert len(scancache.futures) == 12
    assert scancache.next_index == 0
    
    files = [H5File(TEST_PATH / f"{TEST_TIMESTAMP}_sb{subband:02d}.hdf5") for subband in range(16)]
    scan, scan_futures = scancache.get_scan_from_file(files[0], None)
    
    assert scancache.next_index == 1
    assert isinstance(scancache.scans[0], Scan)
    assert scan is scancache.scans[0]

    for file in files[1:]:
        scan, scan_futures = scancache.get_scan_from_file(file, None)
        assert scan is scancache.scans[0]
    
    assert all(file is not None for file in scan.files)
    
    scancache.remove(scan)
    assert scancache.scans[0] is None
