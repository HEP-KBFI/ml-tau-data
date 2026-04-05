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

# Short name for each dataset, looked up at DAG-construction time.
# Used as the filename prefix in split outputs (e.g. z_train.parquet).
SHORT_NAMES = {ds: cfg["short_name"] for ds, cfg in DATASETS.items()}

# Identify which datasets are signal and which are background from config.
SIG_DATASET = next(ds for ds, cfg in DATASETS.items() if cfg["is_signal"])
BKG_DATASET = next(ds for ds, cfg in DATASETS.items() if not cfg["is_signal"])

# Single shared directory for weight matrices and bin-edge arrays.
WEIGHTS_DIR = f"{OUTPUT_DIR}/weights"

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

def input_files(dataset):
    """Return the list of raw input ROOT files for a dataset.

    Two supported modes (set in config.yaml):
      - file_list: path to a plain-text file, one path per line
      - input_dir + file_pattern: glob all matching files in a directory

    Sorting the glob result makes the order deterministic across runs,
    which matters for reproducible train/val/test splits.
    """
    cfg = DATASETS[dataset]
    if "file_list" in cfg:
        # Read pre-made list — useful when the files are spread across
        # multiple directories or when an external script produced the list.
        return Path(cfg["file_list"]).read_text().splitlines()
    return sorted(Path(cfg["input_dir"]).glob(cfg.get("file_pattern", "*.root")))


def ntupelized_files_for(dataset):
    """Return all expected ntupelized output paths for a given dataset.

    Used in the per-dataset merge_and_split rules.  Snakemake calls it at
    DAG-construction time to learn which files must exist before the merge
    step can start.

    The output path mirrors the input stem so each job is unambiguously
    traceable back to its source file:
        /scratch/.../run_001.root  →  <temp_dir>/<dataset>/<stem>.parquet
    """
    return [
        f"{TEMP_DIR}/{dataset}/{Path(f).stem}.parquet"
        for f in input_files(dataset)
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
localrules: all, compute_weights, validation


rule all:
    input:
        [f"{OUTPUT_DIR}/{SHORT_NAMES[ds]}_{split}.parquet" for ds in DATASETS for split in SPLITS],
        f"{OUTPUT_DIR}/validation/.done",
        f"{WEIGHTS_DIR}/sig_weights.npy",


# ── stage 1 : ntupelize (one SLURM job per input file) ────────────────────────
# This rule is instantiated once per (dataset, stem) combination.
# Snakemake figures out which instantiations are needed by seeing that
# ntupelized_files() (used as input to merge_and_split) expects paths of the
# form results/<dataset>/ntupelized/<stem>.root, and then looks for a rule
# whose output pattern matches — which is this one.
rule ntupelize:
    input:
        # Lambda functions receive a 'wildcards' object whose attributes are
        # the values captured from the output path of the rule that requested
        # this file.  Here wc.dataset and wc.stem come from the output pattern
        # f"{OUTPUT_DIR}/{dataset}/ntupelized/{stem}.root".
        # We scan the full input file list for this dataset and return the one
        # file whose stem (filename without extension) matches.
        lambda wc: next(
            f for f in input_files(wc.dataset)
            if Path(f).stem == wc.stem
        )
    output:
        # {dataset} and {stem} are the wildcards — Snakemake fills them in
        # for each individual job.  The output path is what triggers the rule:
        # any rule that lists a path matching this pattern as its input will
        # cause this rule to run first.
        # temp() tells Snakemake to delete this file once merge_and_split has
        # consumed it — keeping the per-file parquets around permanently would
        # waste significant disk space.
        temp(f"{TEMP_DIR}/{{dataset}}/{{stem}}.parquet")
    params:
        # params are evaluated at job-submission time (not at DAG-build time).
        # Using a lambda here lets us look up the per-dataset value from config
        # using the concrete wildcard that was resolved for this specific job.
        is_signal = lambda wc: DATASETS[wc.dataset]["is_signal"],
        container = CONTAINER,
    resources:
        # These values are passed to the SLURM profile (profiles/slurm/config.yaml)
        # via {resources.mem_mb} and {resources.runtime} in the sbatch template.
        mem_mb  = 8_000,
        runtime = 120,   # minutes
    shell:
        # ntupelize.py takes single input/output paths directly as Hydra overrides.
        # {input}, {output}, {params.*} are substituted by Snakemake before
        # the string is handed to bash.
        """
        mkdir -p $(dirname {output})
        {params.container} python ntupelizer/scripts/ntupelize.py \
            ++input_path={input} \
            ++output_path={output} \
            ++is_signal={params.is_signal}
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
            mem_mb  = 16_000,
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
        pt_edges = f"{WEIGHTS_DIR}/pt_edges.npy",
        th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
    params:
        output_dir     = WEIGHTS_DIR,
        produce_plots  = config.get("weights", {}).get("produce_plots", False),
        n_files        = config.get("weights", {}).get("n_files_per_sample", -1),
        container      = CONTAINER,
    resources:
        mem_mb  = 8_000,
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
# One rule per (dataset, split) because the output filename uses short_name
# which is a config value, not a Snakemake wildcard.
for _ds, _cfg in DATASETS.items():
    _short = SHORT_NAMES[_ds]
    for _split in SPLITS:
        rule:
            name: f"apply_weights_{_ds}_{_split}"
            localrule: True
            input:
                split    = f"{OUTPUT_DIR}/{_ds}/split/{_short}_{_split}.parquet",
                sig_w    = f"{WEIGHTS_DIR}/sig_weights.npy",
                bkg_w    = f"{WEIGHTS_DIR}/bkg_weights.npy",
                pt_edges = f"{WEIGHTS_DIR}/pt_edges.npy",
                th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
            output:
                f"{OUTPUT_DIR}/{_short}_{_split}.parquet"
            params:
                weights_dir = WEIGHTS_DIR,
                is_signal   = _cfg["is_signal"],
                container   = CONTAINER,
            resources:
                mem_mb  = 8_000,
                runtime = 30,
            shell:
                """
                {params.container} python ntupelizer/scripts/apply_weights.py \
                    -i {input.split} \
                    -w {params.weights_dir} \
                    -o {output} \
                    --is-signal {params.is_signal}
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
        mem_mb  = 8_000,
        runtime = 60,
    shell:
        """
        mkdir -p {params.outdir}
        {params.container} python ntupelizer/scripts/validate_ntuples.py \
            -s {params.sig_file} \
            -b {params.bkg_file} \
            -o {params.outdir}
        """
