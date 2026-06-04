# ml-tau-data

Data processing pipeline for the machine-learned hadronically-decaying tau lepton reconstruction and identification project. Takes EDM4HEP/PodioROOT simulation files and produces flat Parquet ntuples ready for ML training.

## Setup
```
https://github.com/HEP-KBFI/ml-tau-data
git submodule update --init --recursive
```

## Overview

The workflow is managed by **Snakemake** and consists of four stages:

1. **ntupelize** — process each input ROOT file into a per-file Parquet (one SLURM job per file, grouped 20 per job)
2. **merge\_and\_split** — merge all per-file Parquets for each dataset and split into `train` / `test`
3. **weights** — compute `(p, theta)` reweighting matrices from the signal train set, then apply them to every split
4. **validation** — produce summary plots comparing signal and background distributions

Final outputs land in `output_dir` (configured in `ntupelizer/config/workflow.yaml`):

```
<output_dir>/
  z_train.parquet        # signal train (weighted)
  z_test.parquet         # signal test  (weighted)
  qq_train.parquet       # background train (weighted)
  qq_test.parquet        # background test  (weighted)
  weights/               # weight matrices and bin edges
  validation/            # validation plots
```

Intermediate per-file Parquets are written to `temp_dir` and deleted automatically after merging.

Note: Input simulation files can be generated using the scripts in the `sim/` directory (see [Simulation](#simulation) below).

## Setup

Create a virtual environment and install the package with all dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[full]"
```

> **Note:** Snakemake 7.x is required. If you see `AttributeError: module 'pulp' has no attribute 'list_solvers'`, your `pulp` version is incompatible. Fix with:
> ```bash
> pip install "snakemake>=7,<8" "pulp>=2.7,<3"
> ```

## Configuration

Edit `ntupelizer/config/workflow.yaml` before running:

```yaml
output_dir: /path/to/output          # where final Parquets are written
temp_dir:   /path/to/tmp             # scratch space for per-file Parquets

datasets:
  p8_ee_Z_tautau_ecm91:
    input_dir: /path/to/signal/root/
    file_pattern: "*.root"
    short_name: z
    is_signal: true
    train_frac: 0.70

  p8_ee_Z_qq_ecm91:
    input_dir: /path/to/bkg/root/
    file_pattern: "*.root"
    short_name: qq
    is_signal: false
    train_frac: 0.70

weights:
  produce_plots: true
  add_weights: true
```

Ntupelizer parameters (collections, branches, lifetime variables) are in `ntupelizer/config/ntupelizer_base/new.yaml`.

## Running the workflow

**With SLURM** (ntupelize stage runs on the cluster; everything else runs locally):

```bash
snakemake --profile ntupelizer/config/slurm
```

SLURM jobs are submitted to partition `main9`. Each group of 20 ntupelize jobs shares one `sbatch` allocation. Logs are written to `logs/slurm/`.

**Locally** (all stages on the current machine):

```bash
snakemake -j12    # 12 parallel jobs
```

## Repository structure

```
Snakefile                          # workflow definition (all four stages)
ntupelizer/
  config/
    workflow.yaml                  # dataset paths, output dirs, weight settings
    ntupelizer.yaml                # selects ntupelizer variant (new/old)
    ntupelizer_base/new.yaml       # EDM4HEP/PodioROOT ntupelizer config
    slurm/config.yaml              # Snakemake SLURM profile
  scripts/
    ntupelize.py                   # stage 1 entry point (Hydra)
    merge_files.py                 # stage 2 entry point
    compute_weights.py             # stage 3a
    apply_weights.py               # stage 3b
    validate_ntuples.py            # stage 4
    slurm_status.py                # Snakemake cluster-status helper
  tools/
    ntupelizing.py                 # PodioROOTNtuplelizer / EDM4HEPNtupelizer
    clustering.py                  # reco and gen jet clustering (FastJet)
    matching.py                    # reco↔gen jet matching
    gen_tau_info_matcher.py        # MC tau decay-mode and visible p4 extraction
    particle_filters.py            # reco and MC particle selection
    lifetime.py                    # track impact-parameter / lifetime variables
    tau_decaymode.py               # decay mode classification
    weight_tools.py                # (p, theta) reweighting utilities
    general.py                     # shared helpers and DUMMY_P4_VECTOR
sim/                               # Generation and simulation scripts
  cld/                             # CLD detector simulation (FCC-ee)
    CLDConfig/                     # Submodule for CLD configuration
    run_sim.sh                     # SLURM script for CLD gen-sim-reco
  clic/                            # CLIC detector simulation
    CLICPerformance/               # Submodule for CLIC configuration
    run_sim.sh                     # SLURM script for CLIC gen-sim-reco
```

## Simulation

Standalone scripts for generating EDM4HEP/PodioROOT files using **Key4hep** are provided in the `sim/` directory. These scripts handle the full generation-simulation-reconstruction chain:
1. **Generation** — Pythia8 events
2. **Simulation** — Geant4 via `ddsim`
3. **Reconstruction** — Detector-specific reconstruction (Gaudi-based)

The simulation scripts are designed to run as SLURM jobs and require access to `/cvmfs/sw.hsf.org`.

### CLD Simulation (FCC-ee)
```bash
cd sim/cld
sbatch run_sim.sh <sample_name> <seed>
```
Sample names (e.g., `p8_ee_Z_tautau_ecm91`) correspond to Pythia cards in `sim/cld/CLDConfig/pythia/`.

### CLIC Simulation
```bash
cd sim/clic
sbatch run_sim.sh <sample_name> <seed>
```
Sample names (e.g., `p8_ee_qq_ecm380`) correspond to Pythia cards in `sim/clic/pythia/`.

## Container

All heavy processing runs inside an Apptainer container:

```
/home/software/singularity/pytorch.simg:2025-09-01
```

The container is invoked automatically by Snakemake. No manual setup is needed.
