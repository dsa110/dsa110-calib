from setuptools import setup

setup(name='dsa110-calib',
      version='0.3',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsacalib'],
      package_data={'dsacalib':['data/*']},
      requirements=['casatools','astropy','scipy',
                    'numba','dsa110-antpos','h5py',
                    'matplotlib'],
      zip_safe=False)
