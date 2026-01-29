#!/bin/bash

apptainer exec -B /scratch/persistent,/local,/usr/bin,/usr/lib64/slurm,/etc/slurm --env PYTHONPATH=`pwd`:`pwd`/ntupelizer /home/software/singularity/pytorch.simg\:2025-09-01 "$@"