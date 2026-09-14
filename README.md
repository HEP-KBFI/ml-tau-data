# ml-tau-data

Data processing pipeline for the machine-learned hadronically-decaying tau lepton reconstruction and identification project. Takes EDM4HEP/PodioROOT simulation files and produces flat Parquet ntuples ready for ML training.

## Setup
```
https://github.com/HEP-KBFI/ml-tau-data
git submodule update --init --recursive
```

## Overview

The workflow is managed by **Snakemake** and consists of five stages:

1. **ntupelize** — process each input ROOT file into a per-file Parquet (one SLURM job per file, grouped 20 per job)
2. **weights** — accumulate the `(p, theta)` reweighting matrices. Only the `gen_jet_p4` column is read, so this runs directly on the ntupelized batches, before any merging
3. **merge\_and\_split** — stream the ntupelized batches of each dataset into `train` / `test` chunk files of `chunk_size` events, weighting them as they are filled
4. **validation** — produce summary plots comparing signal and background distributions
5. **preprocess\_torch** — convert each chunk into a pre-built `.pt` tensor file

Stage 3 reads each event once and writes it once. It pulls one row group at a
time from the batches, assigns each event to train or test, and appends it to
that split's fill buffer; a buffer writes an output file as soon as it holds
`chunk_size` events, and when an input file runs out the next one is opened and
keeps filling the file that is still open. So a split is never materialised as
one big file, never as an unweighted copy, and never as a merged intermediate,
and memory is bounded by one output file (~0.75 GB at 100k jets) rather than by
the size of the sample.

Two consequences of streaming are worth knowing:

- **The shuffle is local.** Events are mixed within a fill buffer, i.e. within
  an output file, not across the whole sample. The inputs are independent
  simulation runs of the same process, so this only matters if the training
  loader relies on the file order itself being random. The train/test
  assignment *is* global: it is drawn so that the test events are a uniformly
  random subset of the whole sample and the split sizes land exactly on
  `train_frac`, which is what the parquet-footer pre-scan at the start is for.
- **The weight histograms cover each whole sample**, not only its train split,
  since they are built before the split exists. The split is random, so the two
  distributions are statistically the same; restricting them to the train split
  would mean materialising that split first.

Final outputs land in `output_dir` (configured in `ntupelizer/config/workflow.yaml`):

```
<output_dir>/
  z_train_00000.parquet  # signal train (weighted), <= chunk_size events
  z_train_00001.parquet  # ... as many files as the split needs
  z_train_00000.pt       # pre-built tensors for z_train_00000.parquet
  z_test_00000.parquet   # signal test  (weighted)
  qq_train_00000.parquet # background train (weighted)
  qq_test_00000.parquet  # background test  (weighted)
  weights/               # weight matrices and bin edges
  validation/            # validation plots
  .markers/              # Snakemake completion markers (see below)
```

Every split is a numbered series of files starting at `_00000`. The index is
zero-padded to five digits, so a plain lexicographic listing is already in chunk
order. Set `chunk_size` in `workflow.yaml` to change the events per file
(default 100000).

`row_group_size` (default 1024) sets the parquet row group size inside those
files — about 98 row groups per file at the default `chunk_size`. Row groups are
the unit of a partial read, so small ones keep the dataloader's reads cheap. The
trade-off is bulk reads: measured on a 100k-jet file, 1024 rows per group reads
in full 5.4× slower than 20000 rows per group (1.13 s vs 0.21 s), is ~7% bigger
on disk and carries a 375 KB rather than 25 KB footer. Raise it for a dataset
that is mostly read end to end.

Because the number of chunks in a split is only known once the merge has counted
the events, the chunked stages cannot list their output files up front. They
declare a marker file under `<output_dir>/.markers/` instead — for example
`z_train.chunks` — and the downstream stages glob the chunks at run time. The
practical consequence is that deleting a single chunk `.parquet` does not make
Snakemake rebuild it; delete the corresponding marker to force the split to be
rewritten.

The ntupelized per-batch Parquets under `temp_dir` are the pipeline's only
intermediate, and they are kept rather than deleted. They cost about one extra
copy of the dataset, and in exchange stages 2–5 can be re-run on their own: a new
`chunk_size`, a different `train_frac` or recomputed weights all replay from the
batches in minutes instead of re-ntupelizing thousands of ROOT files. Delete
`temp_dir` by hand once a dataset is final.

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
> pip install "snakemake>=7,<8" "pulp>=2.7,<2.8"
> ```

## Configuration

Edit `ntupelizer/config/workflow.yaml` before running:

```yaml
output_dir: /path/to/output          # where final Parquets are written
temp_dir:   /path/to/tmp             # scratch space for per-file Parquets
chunk_size: 100000                   # max events per output Parquet file

ntupelizer_class: DecayProductNtupelizer   # adds the gen_jet_tau_vis_daughter_* fields

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

`ntupelizer_class` selects what the ntuples contain: `PodioROOTNtuplelizer` for the
standard jet-level ntuples, or `DecayProductNtupelizer` for the ParTauDETR
(tau daughter) dataset, which additionally fills `gen_jet_tau_vis_daughter_p4s`,
`gen_jet_tau_vis_daughter_pdgs` and `gen_jet_tau_vis_daughter_charges` per gen jet.

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

## ALEPH data

The ALEPH ntuples are produced by a separate, self-contained pipeline under
`ntupelizer/aleph/`, configured in `ntupelizer/aleph/config/config.yaml`. It runs
at jet level (`output_level: jet`), one row per jet, which is what the chunked
dataset is built from; `output_level: event` gives one row per event instead,
with the jets of the event in list columns.

Its output is chunked the same way as the main workflow's: numbered files of
`jets_per_file` jets (default 100000), written with `row_group_size` rows per row
group. At jet level that is exactly 100000 rows per file. Rows are never split
across files, so an event-level dataset instead lands on a whole-event boundary
just under the target.

Write and submit the processing jobs:

```bash
python3 ntupelizer/aleph/scripts/ntupelize_all.py    # writes submission_scripts/chunk_*.sh
for job in <output_dir>/submission_scripts/chunk_*.sh; do sbatch "$job"; done
```

Each job streams the ROOT files assigned to it straight into chunk files, so
there is no per-ROOT-file intermediate. The jobs run concurrently and cannot
share a file index, so their filenames carry a per-job prefix
(`job0000_00000.parquet`) and each job's last file holds only its leftovers. To
turn those tails into one uniform series — or to rechunk a dataset that was
produced before this was in place — run:

```bash
python3 ntupelizer/aleph/scripts/rechunk.py \
    -i /local/laurits/ALEPH/ALEPH_jet \
    -o /local/laurits/ALEPH/ALEPH_jet_chunked
```

`rechunk.py` streams a row group at a time and writes each output file once, so
its memory use is bounded by one output file. It reads the level (jet or event)
off the input schema, refuses to overwrite its own inputs, and fails rather than
reporting success if the row count in does not match the row count out. Pass
`--jets-per-file`, `--row-group-size` or `--prefix` to override the defaults.

## Repository structure

```
Snakefile                          # workflow definition (all five stages)
ntupelizer/
  config/
    workflow.yaml                  # dataset paths, output dirs, weight settings
    ntupelizer.yaml                # selects ntupelizer variant (new/old)
    ntupelizer_base/new.yaml       # EDM4HEP/PodioROOT ntupelizer config
    slurm/config.yaml              # Snakemake SLURM profile
  scripts/
    ntupelize.py                   # stage 1 entry point (Hydra)
    merge_files.py                 # stage 3 entry point (merge/split/weight/chunk)
    compute_weights.py             # stage 2
    apply_weights.py               # standalone: re-weight existing chunks
    validate_ntuples.py            # stage 4
    preprocess_torch.py            # stage 5
    slurm_status.py                # Snakemake cluster-status helper
  aleph/                           # standalone ALEPH pipeline (see above)
    scripts/ntupelize_all.py       # writes the per-chunk SLURM job scripts
    scripts/ntupelize_list.py      # one job: ROOT files -> ~100k-jet chunks
    scripts/rechunk.py             # rechunk existing .parquet into ~100k-jet files
    tools/chunking.py              # chunked parquet writer shared by the two
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
