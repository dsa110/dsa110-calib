from setuptools import setup

setup(name='dsa110-calib',
      version='0.3',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsacalib'],
      package_data={'dsacalib':['data/*']},
      install_requires=['astropy==4.0',
                        'casatools==6.0.0.27',
                        'casatasks==6.0.0.27',
                        'cython==0.29.14',
                        'h5py==2.10',
                        'matplotlib==2.2.4',
                        'numba==0.47.0',
                        'numpy==1.17.5',
                        'pandas==0.25.3',
                        'psrdada-python', 
                        'pytest==5.3.2',
                        'scipy==1.4.1',
                        'dsa110-antpos'
                        ],
      dependency_links = [
          "https://github.com/dsa110/dsa110-antpos/tarball/master#egg=dsa110-antpos-0",
          "https://github.com/AA-ALERT/psrdada-python/tarball/master#egg=psrdada-python-0",
          "https://casa-pip.nrao.edu/repository/pypi-casa-release/simple/casatools",
          "https://casa-pip.nrao.edu/repository/pypi-casa-release/simple/casatasks"])
