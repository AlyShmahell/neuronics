SHELL := /bin/bash
CC := g++
TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf;print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf;print(" ".join(tf.sysconfig.get_link_flags()))')
build:
	g++ -std=c++11 -shared shannon/shannon.cc -o shannon/shannon.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2
test:
	python3 tests.py
test_setup:
	python3 -m virtualenv env
	source env/bin/activate && which python && pip3 install .