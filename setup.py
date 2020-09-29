from setuptools import setup

setup(name='dsa110-calib',
      version='0.3',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsacalib'],
      package_data={'dsacalib':['data/*',
                                'data/template_gcal_ant/*',
                                'data/template_gcal_ant/ANTENNA/*',
                                'data/template_gcal_ant/FIELD/*',
                                'data/template_gcal_ant/HISTORY/*',
                                'data/template_gcal_ant/OBSERVATION/*',
                                'data/template_gcal_ant/SPECTRAL_WINDOW/*'
      ]},
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
                        'dsa110-pyutils',
                        'dsa110-meridian-fs'
      ],
      dependency_links = [
          "https://github.com/dsa110/dsa110-antpos/tarball/master#egg=dsa110-antpos",
          "https://github.com/dsa110/dsa110-pyutils/tarball/master#egg=dsa110-pyutils",
          "https://github.com/dsa110/dsa110-meridian-fs/tarball/development#egg=dsa110-meridian-fs"
          ]
)
