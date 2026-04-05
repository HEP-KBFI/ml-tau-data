#!/bin/bash
# Run the ML-Tau Snakemake workflow locally or on SLURM.
# Must be run from the repo root.
# Usage:
#   ./ntupelizer/scripts/run_workflow.sh            # local, 1 core
#   ./ntupelizer/scripts/run_workflow.sh -j 8       # local, 8 cores
#   ./ntupelizer/scripts/run_workflow.sh --slurm    # SLURM cluster
#   ./ntupelizer/scripts/run_workflow.sh -n         # dry-run

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
SNAKEFILE="Snakefile"
CONFIG="ntupelizer/config/workflow.yaml"
CORES=1
SLURM=false
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --slurm) SLURM=true ;;
        *)       EXTRA_ARGS+=("$arg") ;;
    esac
done

mkdir -p logs/slurm

if $SLURM; then
    "$PYTHON_BIN" ntupelizer/scripts/snakemake_runner.py \
        --snakefile "$SNAKEFILE" \
        --configfile "$CONFIG" \
        --profile ntupelizer/config/slurm \
        "${EXTRA_ARGS[@]}"
else
    "$PYTHON_BIN" ntupelizer/scripts/snakemake_runner.py \
        --snakefile "$SNAKEFILE" \
        --configfile "$CONFIG" \
        --cores "$CORES" \
        "${EXTRA_ARGS[@]}"
fi
