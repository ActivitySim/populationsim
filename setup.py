from ez_setup import use_setuptools
use_setuptools()  # nopep8

from setuptools import setup, find_packages

setup(
    name='populationsim',
    version='0.5.1',
    description='Population Synthesis',
    author='contributing authors',
    author_email='ben.stabler@rsginc.com',
    license='BSD-3',
    url='https://github.com/ActivitySim/populationsim',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    include_package_data=True,
    install_requires=[
        'activitysim >= 0.9.9.1',
        'numpy >= 1.16.1',
        'pandas >= 1.1.0',
        'ortools >= 5.1.4045'
    ]
)
