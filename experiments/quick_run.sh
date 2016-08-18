#!/usr/local/bin/bash

cuda=/opt/cuda-7.5

cuDNNv4=/u/ebanner/builds/cudnn-7.0-linux-x64-v4.0-prod
cuDNNv5=/u/ebanner/builds/cudnn-7.5-linux-x64-v5.0-ga

cuDNN=$cuDNNv5

export LD_LIBRARY_PATH=$cuDNN/lib64:$cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$cuDNN/include:$CPATH
export LIBRARY_PATH=$cuDNN/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=$cuDNN

export CUDA_HOME=$cuda

cd code

THEANO_FLAGS=device=cpu,optimizer=fast_compile python train.py $@
