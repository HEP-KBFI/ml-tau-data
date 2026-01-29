#!/usr/bin/env python
"""
CLI script to run the ntupelizer law workflow.

Usage:
    # Run with local scheduler (no central scheduler needed)
    python run_workflow.py NtupelizeSample --sample-name p8_ee_Z_Ztautau_ecm380 --local-scheduler

    # Run on SLURM (default workflow)
    python run_workflow.py NtupelizeSample --sample-name p8_ee_Z_Ztautau_ecm380 --workflow slurm

    # Run locally (for testing)
    python run_workflow.py NtupelizeSample --sample-name p8_ee_Z_Ztautau_ecm380 --workflow local

    # Customize SLURM parameters
    python run_workflow.py NtupelizeSample --sample-name p8_ee_Z_Ztautau_ecm380 \\
        --workflow slurm \\
        --slurm-partition gpu \\
        --slurm-walltime 08:00:00 \\
        --slurm-memory 16000

    # Run the full pipeline with old config (ntupelizer.yaml)
    python run_workflow.py FullPipeline --config-type old --workflow slurm

    # Ntupelize all samples
    python run_workflow.py NtupelizeAllSamples --workflow slurm

    # Validate a sample (runs locally after ntupelization)
    python run_workflow.py ValidateSample --sample-name p8_ee_Z_Ztautau_ecm380

    # Run with multiple workers (local workflow)
    python run_workflow.py NtupelizeSample --sample-name p8_ee_Z_Ztautau_ecm380 --workflow local --workers 4

Config types:
    old: ntupelizer.yaml (legacy LCIO-based ntupelizer)
    new: podio_root_ntupelizer.yaml (new Podio ROOT-based ntupelizer) [default]

Workflow types:
    slurm: Submit jobs to SLURM cluster (default for NtupelizeSample)
    local: Run jobs locally (for testing or small datasets)
"""

import os
import sys

# Add workspace root to path for imports (so ntupelizer package is importable)
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, workspace_root)


def main():
    """Run law workflow from command line."""
    import law
    import luigi

    # Import tasks to register them with law
    from ntupelizer.orchestration_.tasks import (
        NtupelizeFile,
        NtupelizeSample,
        NtupelizeAllSamples,
        ValidateSample,
        ValidateAllSamples,
        CompareValidation,
        FullPipeline,
    )

    # Run luigi with command line arguments
    luigi.run()


if __name__ == "__main__":
    main()
