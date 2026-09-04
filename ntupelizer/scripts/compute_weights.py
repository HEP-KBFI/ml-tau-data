"""Compute pT/theta weight matrices for signal and background train splits.

Bins are read from the Hydra config 'weighting' (ntupelizer/config/weighting.yaml).
Operational parameters map directly to the keys under 'weights:' in workflow.yaml.

The weight matrix is only a 2D histogram of (theta, p), so it is accumulated
incrementally by reading the train parquet one row group at a time.  This keeps
memory flat instead of loading the full train split.

Usage:
    compute_weights.py -i <signal_train> -b <background_train> -o <output_dir>
                       [-n <n_files>] [-p]

Options:
    -i <signal_train>      Path to the signal train parquet file.
    -b <background_train>  Path to the background train parquet file.
    -o <output_dir>        Directory where weight matrices (.npy) and optional
                           plots will be written.
    -n <n_files>           Max number of events to load per sample
                           (n_files_per_sample in workflow.yaml). [default: -1]
    -p                     Produce diagnostic weight-matrix plots
                           (produce_plots in workflow.yaml). [default: False]
"""

import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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


def _gen_jet_theta_p(p4_struct):
    """Return (theta_deg, p) arrays from a gen_jet_p4 struct column/array."""
    pt = pc.struct_field(p4_struct, "pt").to_numpy()
    eta = pc.struct_field(p4_struct, "eta").to_numpy()
    theta = np.degrees(2.0 * np.arctan(np.exp(-eta)))
    p = pt * np.cosh(eta)
    return theta, p


def _accumulate_gen_jet_histogram(path, theta_edges, p_edges, n_max=-1):
    """Accumulate the (theta, p) histogram of gen_jet_p4 by streaming batches."""
    pf = pq.ParquetFile(path)
    hist = np.zeros((len(theta_edges) - 1, len(p_edges) - 1), dtype=np.float64)
    n_seen = 0
    for batch in pf.iter_batches(batch_size=1024):
        theta, p = _gen_jet_theta_p(batch.column("gen_jet_p4"))
        if n_max > 0:
            remaining = n_max - n_seen
            if remaining <= 0:
                break
            if remaining < len(theta):
                theta = theta[:remaining]
                p = p[:remaining]
        h, _, _ = np.histogram2d(theta, p, bins=(theta_edges, p_edges))
        hist += h
        n_seen += len(theta)
    return hist


if __name__ == "__main__":
    args = docopt(__doc__)

    signal_train = args["-i"]
    bkg_train = args["-b"]
    output_dir = args["-o"]
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

    # ── accumulate normalised 2-D histograms (theta × pT) incrementally ───────
    sig_hist = _accumulate_gen_jet_histogram(
        signal_train, theta_edges, p_edges, n_files
    )
    bkg_hist = _accumulate_gen_jet_histogram(bkg_train, theta_edges, p_edges, n_files)

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
