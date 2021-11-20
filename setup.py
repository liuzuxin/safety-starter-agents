#!/usr/bin/env python

from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Safety Starter Agents is designed to work with Python 3.6 and greater. " \
    + "Please install it before proceeding."

# setup(
#     name='safe_rl',
#     packages=['safe_rl'],
#     install_requires=[
#         'gym~=0.15.3',
#         'joblib==0.14.0',
#         'matplotlib==3.1.1',
#         'mpi4py==3.0.2',
#         'mujoco_py==2.0.2.7',
#         'numpy~=1.17.4',
#         'seaborn==0.8.1',
#         'tensorflow==1.13.1',
#     ],
# )

setup(
    name='safe_rl',
    packages=['safe_rl'],
    install_requires=[
        'gym~=0.17.2',
        'joblib',
        'matplotlib',
        'mujoco_py==2.0.2.7',
        'numpy',
        'seaborn==0.8.1',
        'tensorflow~=1.15.1',
    ],
)

# mpi4py should be installed via conda
# conda install -c conda-forge mpi4py

# works well
# mpi                       1.0                       mpich    conda-forge
# mpi4py                    2.0.0                    py36_2    conda-forge
# mpich                     3.4.2              h846660c_100    conda-forge

# works bad
# mpi                       1.0                     openmpi    conda-forge
# mpi4py                    3.0.3            py37hd955b32_1
# openmpi                   4.0.2                hb1b8bf9_1