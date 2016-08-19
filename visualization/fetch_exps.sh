#!/bin/bash

dirs='train hyperparams models batch_weights'

rm -rf store

for dir in $dirs
do
    mkdir -p store/$dir
done

for dir in $dirs
do
    scp -r submit64.cs.utexas.edu:/u/ebanner/scratch/code/ICHI-2016-challenge/cnn/store/$dir store
done
