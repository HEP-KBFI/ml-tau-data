"""
Minimal ML-Tau data processing workflow.

Stages:
  1. ntupelize     - process each input file 1:1 to output
  2. weights       - accumulate the (p, theta) weight matrices straight from the
                     ntupelized batches
  3. merge_split   - stream the ntupelized batches of each dataset into
                     train/test chunk files of `chunk_size` events, weighting
                     as they are filled
  4. validation    - produce validation plots
  5. torch         - convert each chunk to a pre-built .pt tensor file

The final products are numbered chunk files in OUTPUT_DIR, e.g.
z_train_00000.parquet, z_train_00001.parquet, ..., qq_test_00000.parquet, plus a
.pt file next to each of them.  A split is never materialised as a single file,
nor as an unweighted copy, nor as a merged intermediate: stage 3 reads each
event once and writes it once, buffering only as much as one output file holds.
"""

import os
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
# Snakemake reads workflow/config.yaml at startup and makes its contents
# available as the global 'config' dict throughout all rules and Python code
# in this file. The path is relative to the working directory where you invoke
# snakemake — which is the repo root when running from there.
configfile: "ntupelizer/config/workflow.yaml"  # dataset-level config; processing config lives in ntupelizer/config/

# Pull the 'datasets' mapping out of config so we can iterate over it.
# Each key is a dataset name (e.g. "z_91gev"); the value is a dict with
# input_dir, is_signal, train_frac, etc. — see workflow/config.yaml.
DATASETS   = config["datasets"]
OUTPUT_DIR = config["output_dir"]  # top-level output directory from workflow.yaml
TEMP_DIR   = config.get("temp_dir", "/tmp/ml_tau_ntupelized")  # deleted after merge

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


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── rule all — the final target ───────────────────────────────────────────────
# Snakemake works backwards from the requested output files to figure out
# which rules to run.  'rule all' lists the ultimate desired outputs so that
# running `snakemake` with no arguments processes every dataset end-to-end.
#
# expand() is a Snakemake helper that produces the cross-product of all
# wildcard values, e.g. for two datasets it yields:
#   [f"{OUTPUT_DIR}/z_91gev/validation.done", f"{OUTPUT_DIR}/qq_91gev/validation.done"]
# Rules listed here always run on the local machine even when --profile slurm
# is active. ntupelize (one job per ROOT file) is the only rule submitted to
# SLURM — everything else is fast enough to run locally.
localrules: all, compute_weights, validation, preprocess_torch


rule all:
    input:
        [chunks_marker(SHORT_NAMES[ds], split) for ds in DATASETS for split in SPLITS],
        f"{MARKER_DIR}/torch.done",
        f"{OUTPUT_DIR}/validation/.done",
        f"{WEIGHTS_DIR}/sig_weights.npy",


# ── stage 1 : ntupelize (one SLURM job per batch of input files) ────────────
# Each job processes BATCH_SIZE ROOT files and merges them into one parquet.
# Using a batch index wildcard instead of a per-file stem wildcard keeps the
# DAG size at O(N_files / BATCH_SIZE) rather than O(N_files), making DAG
# construction fast even with thousands of input files.
rule ntupelize:
    input:
        # Resolve the list of ROOT files for this batch index.
        # _DATASET_BATCHES is pre-computed at startup so this lookup is O(1).
        lambda wc: _DATASET_BATCHES[wc.dataset][int(wc.batch_idx)]
    output:
        # One output parquet per batch, kept rather than temp(): these are the
        # only intermediate the pipeline has, and holding on to them means the
        # merge/split/weight stage can be re-run — with a different chunk_size
        # or train_frac, say — without re-ntupelizing thousands of ROOT files.
        f"{TEMP_DIR}/{{dataset}}/batch_{{batch_idx}}.parquet"
    params:
        is_signal        = lambda wc: DATASETS[wc.dataset]["is_signal"],
        ntupelizer_class = NTUPELIZER_CLASS,
        container        = CONTAINER,
        # Temporary per-job directory for individual per-file parquets that
        # are concatenated into the single batch output at the end.
        per_file_tmp = lambda wc: f"{TEMP_DIR}/{wc.dataset}/.batch_{wc.batch_idx}_tmp",
    resources:
        # Scale memory and runtime with batch size.
        mem_mb  = lambda wc, input: 2_000 * len(input),
        cpus    = 1,
        runtime = lambda wc, input: 20 * len(input),
    shell:
        # Process each file in the batch, then concatenate into one parquet.
        # The per-file tmp directory is cleaned up regardless of success/failure.
        # Individual file failures are non-fatal: a single corrupt/empty ROOT
        # file should not abort the entire batch.  The batch only fails if no
        # per-file parquets were produced at all (caught in the merge step).
        """
        mkdir -p {params.per_file_tmp}
        trap 'rm -rf {params.per_file_tmp}' EXIT

        n_ok=0
        n_fail=0
        for f in {input}; do
            stem=$(basename "$f" .root)
            if {params.container} python ntupelizer/scripts/ntupelize.py \
                    ++input_path="$f" \
                    ++output_path="{params.per_file_tmp}/$stem.parquet" \
                    ++is_signal={params.is_signal} \
                    ++ntupelizer_class={params.ntupelizer_class} \
                    hydra.run.dir=/tmp; then
                n_ok=$((n_ok + 1))
            else
                echo "WARNING: ntupelize failed for $f (exit $?), skipping"
                n_fail=$((n_fail + 1))
            fi
        done
        echo "Batch summary: $n_ok succeeded, $n_fail failed"

        {params.container} python - <<'PYEOF'
import awkward as ak, glob, sys
import pyarrow as pa
import pyarrow.parquet as pq

files = sorted(glob.glob("{params.per_file_tmp}/*.parquet"))
if not files:
    sys.exit("No per-file parquets produced for this batch")

def _as_array(x):
    # ak.from_parquet returns ak.Record for files written with ak.to_parquet(ak.Record(...)).
    # Convert to ak.Array so the schema is consistent regardless of how the file was written.
    if isinstance(x, ak.Record):
        return ak.Array({{k: x[k] for k in x.fields}})
    return x

def _to_plain_table(arr):
    # Strip awkward-specific schema/field metadata so ParquetWriter sees a
    # stable, comparable schema across files.
    table = ak.to_arrow_table(arr, extensionarray=False)
    clean_schema = pa.schema([f.with_metadata(None) for f in table.schema], metadata=None)
    return table.cast(clean_schema)

# Stream-write one file at a time instead of holding all files in memory at
# once.  For large samples (e.g. qq) a batch of 20 files can exceed the job
# memory if we concatenate them all together, so we keep only one file plus the
# open parquet writer in memory here.
first = _as_array(ak.from_parquet(files[0]))
schema = _to_plain_table(first).schema

writer = pq.ParquetWriter(
    "{output}", schema, compression="zstd", compression_level=6, use_byte_stream_split=True
)
try:
    for f in files:
        writer.write_table(_to_plain_table(_as_array(ak.from_parquet(f))))
finally:
    writer.close()
PYEOF
        """


# ── stage 2 : compute weight matrices (single global job) ────────────────────
# Weights are computed by comparing the signal and background (p, theta)
# distributions.  Only the gen_jet_p4 column is read, so this runs directly on
# the ntupelized batches — before any merging — which is what allows stage 3 to
# apply the weights in the same pass that writes its output.
rule compute_weights:
    input:
        sig = lambda wc: ntupelized_files_for(SIG_DATASET),
        bkg = lambda wc: ntupelized_files_for(BKG_DATASET),
    output:
        sig_w    = f"{WEIGHTS_DIR}/sig_weights.npy",
        bkg_w    = f"{WEIGHTS_DIR}/bkg_weights.npy",
        p_edges  = f"{WEIGHTS_DIR}/p_edges.npy",
        th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
    params:
        output_dir     = WEIGHTS_DIR,
        sig_dir        = f"{TEMP_DIR}/{SIG_DATASET}",
        bkg_dir        = f"{TEMP_DIR}/{BKG_DATASET}",
        produce_plots  = config.get("weights", {}).get("produce_plots", False),
        n_files        = config.get("weights", {}).get("n_files_per_sample", -1),
        container      = CONTAINER,
    resources:
        mem_mb  = 32_000,
        runtime = 30,
    shell:
        """
        mkdir -p {params.output_dir}
        {params.container} python ntupelizer/scripts/compute_weights.py \
            -i {params.sig_dir} \
            -b {params.bkg_dir} \
            -o {params.output_dir} \
            -n {params.n_files} \
            $([ '{params.produce_plots}' = 'True' ] && echo '-p' || true)
        """


# ── stage 3 : stream the batches into weighted train/test chunks ─────────────
# One rule is generated per dataset because the output paths include the
# dataset-specific short_name (e.g. z_train_00000.parquet, qq_test_00003.parquet).
# Snakemake output patterns must be statically derivable from wildcards alone,
# so a single parameterised rule cannot encode a config-lookup in its output
# path.  Generating one rule per dataset at DAG-construction time is the
# idiomatic Snakemake solution for this pattern.
#
# Row groups are pulled from the batches in turn and appended to the train or
# test fill buffer, each of which writes an output file once it holds
# chunk_size events; an input running out mid-file just means the next one is
# opened and keeps filling.
#
# How many chunk files a split needs depends on its event count, which is only
# known once the merge has run, so each split is represented by a marker file
# plus its first chunk (which always exists) rather than by every chunk.
for _ds, _cfg in DATASETS.items():
    _short = SHORT_NAMES[_ds]
    # The .pt tensors of stage 5 are built from these chunks, so drop any that
    # were built from a previous run's chunks; the .parquet files themselves are
    # cleaned up by merge_files.py, which owns them.
    _stale_pt = " ".join(
        f"{OUTPUT_DIR}/{_short}_{split}_*.{ext}"
        for split in SPLITS
        for ext in ("pt", "pt.progress")
    )
    rule:
        name: f"merge_and_split_{_ds}"
        localrule: True
        input:
            # Capture _ds in the default arg to avoid the Python late-binding
            # closure problem inside a for-loop.
            batches  = lambda wc, ds=_ds: ntupelized_files_for(ds),
            sig_w    = f"{WEIGHTS_DIR}/sig_weights.npy",
            bkg_w    = f"{WEIGHTS_DIR}/bkg_weights.npy",
            p_edges  = f"{WEIGHTS_DIR}/p_edges.npy",
            th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
        output:
            # One marker per split, plus the first chunk of each, which the
            # validation stage reads directly.
            [touch(chunks_marker(_short, split)) for split in SPLITS],
            [first_chunk(_short, split) for split in SPLITS],
        params:
            input_dir   = f"{TEMP_DIR}/{_ds}",
            output_dir  = OUTPUT_DIR,
            short_name  = _short,
            train_frac  = _cfg.get("train_frac", 0.8),
            chunk_size  = CHUNK_SIZE,
            row_group   = ROW_GROUP_SIZE,
            side        = "sig" if _cfg["is_signal"] else "bkg",
            weights_dir = WEIGHTS_DIR,
            stale_pt    = _stale_pt,
            container   = CONTAINER,
        resources:
            mem_mb  = 32_000,
            runtime = 120,
        shell:
            """
            mkdir -p {params.output_dir}
            rm -f {params.stale_pt}
            {params.container} python ntupelizer/scripts/merge_files.py \
                -i {params.input_dir} \
                -o {params.output_dir} \
                -s {params.short_name} \
                -f {params.train_frac} \
                -w {params.weights_dir} \
                --side {params.side} \
                --chunk-size {params.chunk_size} \
                --row-group-size {params.row_group}
            """


# ── stage 4 : validation ──────────────────────────────────────────────────────
# Single global job comparing the signal and background distributions, including
# the weight distributions that apply_weights.py -p used to produce.
rule validation:
    input:
        # Plots are made from the first train chunk of each dataset: the
        # validate_ntuples.py plots load their input in full, and one chunk
        # (CHUNK_SIZE jets) is plenty of statistics for distribution shapes.
        [first_chunk(SHORT_NAMES[ds], "train") for ds in DATASETS]
    output:
        touch(f"{OUTPUT_DIR}/validation/.done")
    params:
        outdir    = f"{OUTPUT_DIR}/validation",
        sig_file  = first_chunk(SHORT_NAMES[SIG_DATASET], "train"),
        bkg_file  = first_chunk(SHORT_NAMES[BKG_DATASET], "train"),
        container = CONTAINER,
    resources:
        mem_mb  = 32_000,
        runtime = 60,
    shell:
        """
        mkdir -p {params.outdir}
        {params.container} python ntupelizer/scripts/validate_ntuples.py \
            -s {params.sig_file} \
            -b {params.bkg_file} \
            -o {params.outdir}
        """


# ── stage 5 : preprocess to .pt tensors ──────────────────────────────────────
# One-time conversion of the final weighted .parquet chunks into pre-built
# PyTorch tensor files (.pt) so the training dataloader can skip the
# parquet→tensor conversion on every run.  One .pt is written next to each
# chunk (z_train_0.parquet → z_train_0.pt); already converted chunks are
# skipped, so an interrupted job resumes where it left off.
rule preprocess_torch:
    input:
        [chunks_marker(SHORT_NAMES[ds], split) for ds in DATASETS for split in SPLITS]
    output:
        touch(f"{MARKER_DIR}/torch.done")
    params:
        input_dir  = OUTPUT_DIR,
        max_cands  = config.get("max_cands", 20),
        container  = CONTAINER,
    resources:
        mem_mb  = 32_000,
        runtime = 120,
    shell:
        """
        {params.container} python ntupelizer/scripts/preprocess_torch.py \
            -i {params.input_dir} \
            --max-cands {params.max_cands}
        """
