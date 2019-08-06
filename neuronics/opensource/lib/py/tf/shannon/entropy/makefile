TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0
GPUCFLAGS = -c
LFLAGS    =  -shared -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -I ..
GPULFLAGS = -x cu -shared -Xcompiler -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB)
DEBUG = -g -G
GPUDEF    = -D GOOGLE_CUDA=1
CGPUFLAGS = -lcudart


SRC       = entropy_op.cc
GPUSRC    = entropy_op_cu.cc
PROD      = entropy_op.so
GPUPROD   = entropy_op.cu.so

default: gpu

cpu:
	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) -o $(PROD)

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) -o $(GPUPROD) $(GPUDEF) -I/usr/local/ --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) -o $(PROD) $(GPUDEF)

clean:
	rm -f $(PROD) $(GPUPROD)