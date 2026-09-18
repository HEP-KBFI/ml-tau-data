# Shared configuration, derived constants and path helpers.
#
# Included first by the root Snakefile.  Snakemake shares one namespace across
# all included files, so everything defined here is visible to the stage files
# that follow -- they do not import it.

import os
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
# The 'config' dict is populated by the configfile: directive in the root
# Snakefile (and extended by any --configfile / --config given on the command
# line) before this file is included.

# Each key of 'datasets' is a dataset name (e.g. "p8_ee_Z_tautau_ecm91"); the
# value is a dict with input_dir, is_signal, train_frac, etc.
DATASETS   = config["datasets"]
OUTPUT_DIR = config["output_dir"]
TEMP_DIR   = config.get("temp_dir", "/tmp/ml_tau_ntupelized")

# The two output splits we always produce.  Used wherever we need to
# enumerate them (expand(), input lists, etc.)
SPLITS = ["train", "test"]

# Which ntupelizer class to use across the entire workflow.
# Override via workflow.yaml or snakemake --config ntupelizer_class=DecayProductNtupelizer
NTUPELIZER_CLASS = config.get("ntupelizer_class", "PodioROOTNtuplelizer")

# Short name for each dataset, looked up at DAG-construction time.
# Used as the filename prefix in split outputs (e.g. z_train.parquet).
SHORT_NAMES = {ds: cfg["short_name"] for ds, cfg in DATASETS.items()}

# Identify which datasets are signal and which are background from config.
SIG_DATASET = next(ds for ds, cfg in DATASETS.items() if cfg["is_signal"])
BKG_DATASET = next(ds for ds, cfg in DATASETS.items() if not cfg["is_signal"])

# Single shared directory for weight matrices and bin-edge arrays.
WEIGHTS_DIR = f"{OUTPUT_DIR}/weights"

# Maximum number of events (jets) per output .parquet file.  Each split is
# written as a numbered series of files instead of one big file, so that the
# training dataloader can stream / shard them and no single file has to be read
# in full.  Override via workflow.yaml (chunk_size) or --config chunk_size=...
CHUNK_SIZE = config.get("chunk_size", 100_000)

# Rows per parquet row group inside those files.  Row groups are the unit of a
# partial read, so small ones keep the dataloader's reads cheap; see
# DEFAULT_ROW_GROUP_SIZE in merge_files.py for what that costs bulk reads.
ROW_GROUP_SIZE = config.get("row_group_size", 1024)

# Seed the train/test assignment and per-output-file shuffle so reruns with the
# same inputs and configuration produce the same parquet contents.
SPLIT_SEED = config.get("split_seed", None)

# Width of the zero-padded chunk index in the filenames written by
# merge_files.py (CHUNK_INDEX_WIDTH there); keep the two in step.
CHUNK_INDEX_WIDTH = 5

# Index of the chunk the validation plots are made from.  They load their input
# in full, so they look at one chunk (CHUNK_SIZE events) rather than the whole
# split, which is plenty of statistics for distribution shapes.
VALIDATION_CHUNK = 0

# Because the number of chunk files is only known once the merge has counted the
# events, the merge stage cannot declare its outputs one file per rule output.
# It declares one marker file per split here instead, and the downstream stages
# glob the chunks at runtime.  Note the consequence: deleting an individual
# chunk .parquet does not by itself make Snakemake rebuild it — delete the
# corresponding marker (or the whole split) to force a rerun.
MARKER_DIR = f"{OUTPUT_DIR}/.markers"


def chunks_marker(short_name, split):
    """Marker file standing in for the chunk files of one split."""
    return f"{MARKER_DIR}/{short_name}_{split}.chunks"


def first_chunk(short_name, split):
    """Path of the first weighted chunk file of a split — always exists."""
    index = f"{VALIDATION_CHUNK:0{CHUNK_INDEX_WIDTH}d}"
    return f"{OUTPUT_DIR}/{short_name}_{split}_{index}.parquet"


# Number of input ROOT files to process in a single SLURM job.
# Reduces DAG size from O(N_files) to O(N_files / BATCH_SIZE).
BATCH_SIZE = config.get("files_per_job", 20)

# Build the apptainer command prefix once and reuse it in every rule.
# -B mounts host directories into the container so that scripts can read
#    input files (on /scratch) and write output files (on /local, etc.).
# --env PYTHONPATH makes the repo's own packages (ntupelizer/) importable
#    inside the container without needing a full install.
CONTAINER = (
    "apptainer exec"
    " -B /scratch/persistent,/local,/usr/bin,/usr/lib64/slurm,/etc/slurm"
    f" --env PYTHONPATH={os.getcwd()}:{os.getcwd()}/ntupelizer"
    " /home/software/singularity/pytorch.simg:2025-09-01"
)


# ── input discovery ───────────────────────────────────────────────────────────

# Cache file lists under .snakemake_file_lists/ so that the potentially slow
# filesystem glob only runs once (on the first snakemake invocation).
# Delete the directory to force a refresh (e.g. after adding new input files).
_FILE_LIST_CACHE_DIR = ".snakemake_file_lists"


def input_files(dataset):
    """Return the list of raw input ROOT files for a dataset.

    Two supported modes (set in config.yaml):
      - file_list: path to a plain-text file, one path per line
      - input_dir + file_pattern: glob all matching files in a directory
                                   (result cached in .snakemake_file_lists/)

    Sorting the glob result makes the order deterministic across runs,
    which matters for reproducible train/val/test splits.
    """
    cfg = DATASETS[dataset]
    if "file_list" in cfg:
        # Read pre-made list — useful when the files are spread across
        # multiple directories or when an external script produced the list.
        return Path(cfg["file_list"]).read_text().splitlines()

    # Try the local cache first to avoid a slow remote-filesystem glob every run.
    cache_path = os.path.join(_FILE_LIST_CACHE_DIR, f"{dataset}.txt")
    if os.path.exists(cache_path):
        files = Path(cache_path).read_text().splitlines()
        if files:  # never return an empty cache (could be a partial write)
            return files

    # Cache miss: do the (possibly slow) glob and persist the sorted result.
    files = sorted(str(p) for p in Path(cfg["input_dir"]).glob(cfg.get("file_pattern", "*.root")))
    os.makedirs(_FILE_LIST_CACHE_DIR, exist_ok=True)
    Path(cache_path).write_text("\n".join(files))
    return files


# Pre-compute batches at startup so ntupelized_files_for() can return a
# deterministic list of batch-output paths without re-globbing every call.
# Must be placed after input_files() is defined.
_DATASET_BATCHES = {
    ds: [
        _files[i : i + BATCH_SIZE]
        for _files in [list(input_files(ds))]
        for i in range(0, len(_files), BATCH_SIZE)
    ]
    for ds in DATASETS
}


def ntupelized_files_for(dataset):
    """Return all expected ntupelized output paths for a given dataset.

    Each path corresponds to one batch job (batch_NNNNN.parquet), which
    contains the merged output of up to BATCH_SIZE input ROOT files.
    This is consumed by the per-dataset merge_and_split rule.
    """
    n_batches = len(_DATASET_BATCHES[dataset])
    return [
        f"{TEMP_DIR}/{dataset}/batch_{i:05d}.parquet"
        for i in range(n_batches)
    ]
