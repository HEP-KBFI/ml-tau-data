"""Compute pT/theta weight matrices for a signal and a background sample.

Bins are read from the Hydra config 'weighting' (ntupelizer/config/weighting.yaml).
Operational parameters map directly to the keys under 'weights:' in workflow.yaml.

The weight matrix is only a 2D histogram of (theta, p) of the gen jets, so it is
accumulated one input file at a time and only the gen_jet_p4 column is read.
That means it can run directly on the ntupelized per-batch parquets, before the
samples have been merged and split, which is what lets the merge stage apply the
weights in the same pass that writes its output.

Note that the histogram therefore covers the whole sample rather than only its
train split.  The split is a random shuffle, so the two distributions are
statistically the same; computing it from the train split alone would require
materialising that split first, i.e. an extra full pass over the data.

Usage:
    compute_weights.py -i <signal> -b <background> -o <output_dir>
                       [-t <split>] [-n <n_events>] [-p]

Options:
    -i <signal>            Signal input: either a single parquet file or a
                           directory of parquet files, all of which are then
                           histogrammed.
    -b <background>        Background input, same two forms as -i.
    -o <output_dir>        Directory where weight matrices (.npy) and optional
                           plots will be written.
    -t <split>             Consider only the chunks of this split when -i/-b are
                           directories of <short>_<split>_<index>.parquet files.
                           Omit it to use every parquet in the directory.
    -n <n_events>          Max number of events to histogram per sample
                           (n_files_per_sample in workflow.yaml). [default: -1]
    -p                     Produce diagnostic weight-matrix plots
                           (produce_plots in workflow.yaml). [default: False]
"""

import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
from docopt import docopt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Allow importing from the tools directory when running inside the container
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import general as g
import weight_tools as wt


def build_bin_edges(var_cfg):
    """Return numpy linspace bin edges from a weighting.yaml variable block."""
    return np.linspace(var_cfg.range[0], var_cfg.range[1], var_cfg.n_bins + 1)


def _accumulate_gen_jet_histogram(paths, theta_edges, p_edges, n_max=-1):
    """Accumulate the (theta, p) histogram of gen_jet_p4 over a list of files.

    Only the gen_jet_p4 column of one file is held at a time, so memory stays
    flat however many files the sample spans.  n_max, when positive, caps the
    total number of events used across all files.
    """
    hist = np.zeros((len(theta_edges) - 1, len(p_edges) - 1), dtype=np.float64)
    n_seen = 0
    n_files_read = 0
    for path in paths:
        if n_max > 0 and n_seen >= n_max:
            break
        n_files_read += 1
        data = ak.from_parquet(path, columns=["gen_jet_p4"])
        theta, p = wt.gen_jet_theta_p(data)
        if n_max > 0 and n_max - n_seen < len(theta):
            theta = theta[: n_max - n_seen]
            p = p[: n_max - n_seen]
        h, _, _ = np.histogram2d(theta, p, bins=(theta_edges, p_edges))
        hist += h
        n_seen += len(theta)
    print(f"Histogrammed {n_seen} events from {n_files_read} file(s)")
    return hist


if __name__ == "__main__":
    args = docopt(__doc__)

    signal_input = args["-i"]
    bkg_input = args["-b"]
    output_dir = args["-o"]
    split = args["-t"]
    n_files = int(args["-n"]) if args["-n"] else -1
    produce_plots = args["-p"]

    os.makedirs(output_dir, exist_ok=True)

    # Skip computation if the weight matrices already exist.
    sig_out = os.path.join(output_dir, "sig_weights.npy")
    bkg_out = os.path.join(output_dir, "bkg_weights.npy")
    p_edges_out = os.path.join(output_dir, "p_edges.npy")
    theta_edges_out = os.path.join(output_dir, "theta_edges.npy")
    if (
        os.path.exists(sig_out)
        and os.path.exists(bkg_out)
        and os.path.exists(p_edges_out)
        and os.path.exists(theta_edges_out)
    ):
        print(f"Weight matrices already exist in {output_dir}, skipping computation.")
        sys.exit(0)

    # ── load Hydra config for bin edges ───────────────────────────────────────
    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, job_name="compute_weights"):
        cfg = compose(config_name="weighting")

    wcfg = cfg.weighting
    p_edges = build_bin_edges(wcfg.variables.p)
    theta_edges = build_bin_edges(wcfg.variables.theta)

    # ── resolve a directory of inputs (or a single file) ──────────────────────
    sig_paths = g.split_chunk_paths(signal_input, split)
    bkg_paths = g.split_chunk_paths(bkg_input, split)

    # ── accumulate normalised 2-D histograms (theta × pT) incrementally ───────
    sig_hist = _accumulate_gen_jet_histogram(sig_paths, theta_edges, p_edges, n_files)
    bkg_hist = _accumulate_gen_jet_histogram(bkg_paths, theta_edges, p_edges, n_files)

    sig_matrix = sig_hist / np.sum(sig_hist)
    bkg_matrix = bkg_hist / np.sum(bkg_hist)

    # ── compute weight matrices ────────────────────────────────────────────────
    # Signal weights: reweight signal to look like background
    sig_weight_matrix = wt.get_weight_matrix(
        target_matrix=sig_matrix, comp_matrix=bkg_matrix
    )
    # Background weights: reweight background to look like signal
    bkg_weight_matrix = wt.get_weight_matrix(
        target_matrix=bkg_matrix, comp_matrix=sig_matrix
    )

    # ── save matrices ──────────────────────────────────────────────────────────
    np.save(sig_out, sig_weight_matrix)
    np.save(bkg_out, bkg_weight_matrix)

    # Save bin edges alongside so apply_weights.py can reconstruct the lookup
    np.save(p_edges_out, p_edges)
    np.save(theta_edges_out, theta_edges)

    print(f"Saved signal weight matrix  → {sig_out}")
    print(f"Saved background weight matrix → {bkg_out}")

    # ── optional plots ─────────────────────────────────────────────────────────
    if produce_plots:
        wt.visualize_weights_pair(
            sig_matrix=sig_weight_matrix,
            bkg_matrix=bkg_weight_matrix,
            x_bin_edges=p_edges,
            y_bin_edges=theta_edges,
            output_path=os.path.join(output_dir, "weight_matrices.pdf"),
        )
        print("Saved diagnostic plots.")
