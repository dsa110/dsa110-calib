from setuptools import setup

setup(name='dsa110-calib',
      version='0.1',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsacalib'],
      package_data={'dsacalib':['data/*.all','data/templatekcal']}, #,'data/sample.fits']},
      requirements=['casa-python','casa-data','astropy','scipy','numba'],
      zip_safe=False)

