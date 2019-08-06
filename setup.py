import os
from distutils.core import setup, Extension

def pretty(x):
    import re
    return re.sub(r"\n\s+" , "\n" , x)

try:
    import tensorflow as tf
except:
    import sys
    print(pretty(
        f"""
            You need to install tensorflow>=1.12.0 in order to build this package from scratch
            """
    ))
    sys.exit(0)

setup(
    name="neuronics",
    packages=["neuronics"],
    install_requires=['tensorflow==1.12.0'],
    version='0.1.0',
    description='a collection of tensorflow functions.',
    author='Aly Shmahell',
	author_email='aly.shmahell@gmail.com',
    license='Copyrights 2019 Aly Shmahell'
    )

