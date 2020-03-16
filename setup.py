from setuptools import setup

setup(name='dsa110-calib',
      version='0.3',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsacalib'],
      package_data={'dsa110-calib':['dsacalib/data/*.all']},
      requirements=['casa-python','casa-data','astropy','scipy','numba','dsa110-antpos'],
      zip_safe=False)

