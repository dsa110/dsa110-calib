from setuptools import setup
from dsautils.version import get_git_version

setup(name='dsa110-calib',
      version=get_git_version(),
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
      ],
)
