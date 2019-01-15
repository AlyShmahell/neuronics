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

TF_CFLAGS = ' '.join(tf.sysconfig.get_compile_flags())
TF_LFLAGS = ' '.join(tf.sysconfig.get_link_flags())


os.environ["CC"] = "g++"
extra_compile_args = ['-std=c++11', '-shared', '-fPIC', TF_CFLAGS, TF_LFLAGS, '-O2']
extension = Extension(
        name='shannon.so',
        sources=['shannon/shannon.cc'],
        include_dirs=[tf.sysconfig.get_compile_flags()[0].split('-I')[1]],
        extra_compile_args=extra_compile_args,
        language='c++11'
        )


setup(
    name="shannon",
    packages=["shannon"],
    install_requires=['tensorflow'],
    ext_modules=[extension],
    version='0.1.0',
    description='Shannon\'s Information & Entropy Equations implemented as Tensorflow Operations.',
    author='Aly Shmahell',
	author_email='aly.shmahell@gmail.com',
    license='Copyrights 2019 Aly Shmahell'
    )

