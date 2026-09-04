"""
Minimal ML-Tau data processing workflow.

Stages:
  1. ntupelize     - process each input file 1:1 to output
  2. merge_split   - merge per-dataset outputs, split into train/val/test
  3. weights       - compute weights from train, apply to val/test
  4. validation    - produce validation plots per dataset per split
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
        [f"{OUTPUT_DIR}/{SHORT_NAMES[ds]}_{split}.parquet" for ds in DATASETS for split in SPLITS],
        [f"{OUTPUT_DIR}/{SHORT_NAMES[ds]}_{split}.pt" for ds in DATASETS for split in SPLITS],
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
        # One output parquet per batch; temp() deletes it once merge_and_split
        # has consumed it to avoid accumulating large intermediate files.
        temp(f"{TEMP_DIR}/{{dataset}}/batch_{{batch_idx}}.parquet")
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


# ── stage 2 : merge all ntupelized outputs → split files ─────────────────────
# One rule is generated per dataset because the output filenames include the
# dataset-specific short_name (e.g. z_train.parquet, qq_test.parquet).
# Snakemake output patterns must be statically derivable from wildcards alone,
# so a single parameterised rule cannot encode a config-lookup in its output
# path.  Generating one rule per dataset at DAG-construction time is the
# idiomatic Snakemake solution for this pattern.
for _ds, _cfg in DATASETS.items():
    _short = SHORT_NAMES[_ds]
    rule:
        name: f"merge_and_split_{_ds}"
        localrule: True
        input:
            # Capture _ds in the default arg to avoid the Python late-binding
            # closure problem inside a for-loop.
            lambda wc, ds=_ds: ntupelized_files_for(ds)
        output:
            # The f-string resolves _ds and _short at loop time; {{split}}
            # becomes {split} after f-string processing so expand() can fill it.
            [temp(f"{OUTPUT_DIR}/{_ds}/split/{_short}_{split}.parquet") for split in SPLITS]
        params:
            input_dir  = f"{TEMP_DIR}/{_ds}",
            outdir     = f"{OUTPUT_DIR}/{_ds}/split",
            short_name = _short,
            train_frac = _cfg.get("train_frac", 0.8),
            container  = CONTAINER,
        resources:
            mem_mb  = 32_000,
            runtime = 60,
        shell:
            """
            mkdir -p {params.outdir}
            {params.container} python ntupelizer/scripts/merge_files.py \
                -i {params.input_dir} \
                -o {params.outdir} \
                -s {params.short_name} \
                -f {params.train_frac}
            """


# ── stage 3a : compute weight matrices (single global job) ───────────────────
# Weights are computed by comparing signal vs background train distributions.
# One weight matrix is produced for each side (sig_weights.npy, bkg_weights.npy)
# plus the bin-edge arrays so apply_weights can reconstruct the lookup at runtime.
rule compute_weights:
    input:
        sig = f"{OUTPUT_DIR}/{SIG_DATASET}/split/{SHORT_NAMES[SIG_DATASET]}_train.parquet",
        bkg = f"{OUTPUT_DIR}/{BKG_DATASET}/split/{SHORT_NAMES[BKG_DATASET]}_train.parquet",
    output:
        sig_w    = f"{WEIGHTS_DIR}/sig_weights.npy",
        bkg_w    = f"{WEIGHTS_DIR}/bkg_weights.npy",
        p_edges  = f"{WEIGHTS_DIR}/p_edges.npy",
        th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
    params:
        output_dir     = WEIGHTS_DIR,
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
            -i {input.sig} \
            -b {input.bkg} \
            -o {params.output_dir} \
            -n {params.n_files} \
            $([ '{params.produce_plots}' = 'True' ] && echo '-p' || true)
        """


# ── stage 3b : apply weights to every split ──────────────────────────────────
# One rule per split: applies signal and background weight matrices in a single
# call and optionally produces a weight distribution plot.
for _split in SPLITS:
    rule:
        name: f"apply_weights_{_split}"
        localrule: True
        input:
            sig      = f"{OUTPUT_DIR}/{SIG_DATASET}/split/{SHORT_NAMES[SIG_DATASET]}_{_split}.parquet",
            bkg      = f"{OUTPUT_DIR}/{BKG_DATASET}/split/{SHORT_NAMES[BKG_DATASET]}_{_split}.parquet",
            sig_w    = f"{WEIGHTS_DIR}/sig_weights.npy",
            bkg_w    = f"{WEIGHTS_DIR}/bkg_weights.npy",
            p_edges  = f"{WEIGHTS_DIR}/p_edges.npy",
            th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
        output:
            sig_out = f"{OUTPUT_DIR}/{SHORT_NAMES[SIG_DATASET]}_{_split}.parquet",
            bkg_out = f"{OUTPUT_DIR}/{SHORT_NAMES[BKG_DATASET]}_{_split}.parquet",
        params:
            weights_dir   = WEIGHTS_DIR,
            output_dir    = OUTPUT_DIR,
            produce_plots = config.get("weights", {}).get("produce_plots", False),
            container     = CONTAINER,
        resources:
            mem_mb  = 32_000,
            runtime = 30,
        shell:
            """
            {params.container} python ntupelizer/scripts/apply_weights.py \
                -i {input.sig} \
                -b {input.bkg} \
                -w {params.weights_dir} \
                -o {params.output_dir} \
                $([ '{params.produce_plots}' = 'True' ] && echo '-p' || true)
            """


# ── stage 4 : validation ──────────────────────────────────────────────────────
# Single global job taking all weighted files across all datasets and splits.
rule validation:
    input:
        [f"{OUTPUT_DIR}/{SHORT_NAMES[ds]}_train.parquet" for ds in DATASETS]
    output:
        touch(f"{OUTPUT_DIR}/validation/.done")
    params:
        outdir    = f"{OUTPUT_DIR}/validation",
        sig_file  = f"{OUTPUT_DIR}/{SHORT_NAMES[SIG_DATASET]}_train.parquet",
        bkg_file  = f"{OUTPUT_DIR}/{SHORT_NAMES[BKG_DATASET]}_train.parquet",
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
# One-time conversion of the final weighted .parquet files into pre-built
# PyTorch tensor files (.pt) so the training dataloader can skip the
# parquet→tensor conversion on every run.
rule preprocess_torch:
    input:
        [f"{OUTPUT_DIR}/{SHORT_NAMES[ds]}_{split}.parquet" for ds in DATASETS for split in SPLITS]
    output:
        [f"{OUTPUT_DIR}/{SHORT_NAMES[ds]}_{split}.pt" for ds in DATASETS for split in SPLITS]
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
