from setuptools import setup

setup(name='dsa110-calib',
      version='0.1',
      url='http://github.com/dsa110/dsa110-calib/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsacalib'],
      requirements=['casa-python','astropy','scipy'],
      zip_safe=False)
      
