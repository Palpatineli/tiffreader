from setuptools import setup

import subprocess

try:  # Try to create an rst long_description from README.md
    args = "pandoc", "--to", "rst", "README.md"
    long_description = subprocess.check_output(args)
    long_description = long_description.decode()
except Exception as error:
    print("README.md conversion to reStructuredText failed. Error:\n",
          error, "Setting long_description to None.")
    long_description = None

setup(
    name='tiffreader',
    version='0.1.1',
    packages=['tiffreader'],
    url='https://github.com/Palpatineli/tiffreader',
    download_url='https://github.com/Palpatineli/tiffreader/archive/0.1.1.tar.gz',
    license='GPLv3',
    author='Keji Li',
    author_email='mail@keji.li',
    install_requires=['libtiff', 'numpy'],
    tests_require=['pytest'],
    description='convenience wrapper for libtiff',
    long_description=long_description,
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3']
)
