#!/bin/bash

# Theano setup
module load gnu-4.4-compilers
module load fftw-3.3.3
module load perl-5.20.0
module load platform-mpi
module load slurm-14.11.8

CUDAVER="7.5"
module load cuda-${CUDAVER}

export PATH=/shared/apps/cuda${CUDAVER}/bin:$PATH
export LD_LIBRARY_PATH=/shared/apps/cuda${CUDAVER}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ekstrand-abueg.m/.local/lib/:$LD_LIBRARY_PATH
export CUDA_HOME=/shared/apps/cuda${CUDAVER}/

export PYTHONPATH="/home/banner.ed/pico-vectors/preprocess:$PYTHONPATH"

cd code
THEANO_FLAGS=device=cpu,optimizer=fast_compile,mode=FAST_RUN,floatX=float32 python train.py $@
