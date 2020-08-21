from setuptools import setup

setup(name='dsa110-calib',
      version='0.3',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsacalib'],
      package_data={'dsacalib':['data/*']},
      install_requires=['astropy',
                        'casatools',
                        'casatasks',
                        'casadata',
                        'cython',
                        'h5py',
                        'matplotlib',
                        'numba',
                        'numpy',
                        'pandas',
                        'pytest',
                        'codecov',
                        'coverage',
                        'pyyaml',
                        'scipy',
                        'etcd3',
                        'structlog',
                        'dsa110-antpos',
                        'dsa110-pyutils'
      ],
      dependency_links = [
          "https://github.com/dsa110/dsa110-antpos/tarball/master#egg=dsa110-antpos-0",
          "https://github.com/dsa110/dsa110-pyutils/tarball/ds/dev#egg=dsa110-pyutils-0",
          ]
)
