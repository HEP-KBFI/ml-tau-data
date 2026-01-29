#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


apptainer exec -B /scratch/persistent,/local --env PYTHONPATH=`pwd`:`pwd`/ntupelizer /home/software/singularity/pytorch.simg\:2025-09-01 "$@"
