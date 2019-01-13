from distutils.core import setup, Extension
import tensorflow as tf

TF_CFLAGS = ' '.join(tf.sysconfig.get_compile_flags())
TF_LFLAGS = ' '.join(tf.sysconfig.get_link_flags())


extra_compile_args = ['-std=c++11', '-shared', '-fPIC', TF_CFLAGS, TF_LFLAGS, '-O2']
extension = Extension(
        name='shannon.so',
        sources='shannon/shannon.cc',
        library_dirs=['shannon'],
        libraries=['format_id.h'],
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

