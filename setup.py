from ez_setup import use_setuptools
use_setuptools()  # nopep8

from setuptools import setup, find_packages

setup(
    name='populationsim',
    version='0.4.1',
    description='Population Synthesis',
    author='contributing authors',
    author_email='ben.stabler@rsginc.com',
    license='BSD-3',
    url='https://github.com/ActivitySim/populationsim',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    include_package_data=True,
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    install_requires=[
        'activitysim >= 0.9.2',
        'numpy >= 1.16.1',
        'pandas >= 0.24.1',
        'ortools >= 5.1.4045, < 7.5',
        'future >= 0.16.0'
    ]
)
