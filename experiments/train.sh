#!/usr/local/bin/bash

# This script is a wrapper around the python training script.
#
# Call this script from condor!


# added by Anaconda2 2.4.0 installer
export PATH="/u/ebanner/.anaconda2/bin:$PATH"

export PYTHONPATH="/u/ebanner/builds/keras:$PYTHONPATH"

source activate py27

cuda=/opt/cuda-7.5

cuDNNv4=/u/ebanner/builds/cudnn-7.0-linux-x64-v4.0-prod
cuDNNv5=/u/ebanner/builds/cudnn-7.5-linux-x64-v5.0-ga

cuDNN=$cuDNNv5

export LD_LIBRARY_PATH=$cuDNN/lib64:$cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$cuDNN/include:$CPATH
export LIBRARY_PATH=$cuDNN/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=$cuDNN

export CUDA_HOME=$cuda

cd code && python train.py $@

source deactivate
