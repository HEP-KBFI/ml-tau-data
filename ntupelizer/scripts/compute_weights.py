"""Compute pT/theta weight matrices for signal and background train splits.

Bins are read from the Hydra config 'weighting' (ntupelizer/config/weighting.yaml).
Operational parameters map directly to the keys under 'weights:' in workflow.yaml.

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
import numpy as np
import awkward as ak
from pathlib import Path
from docopt import docopt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Allow importing from the tools directory when running inside the container
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt
import general as g


def build_bin_edges(var_cfg):
    """Return numpy linspace bin edges from a weighting.yaml variable block."""
    return np.linspace(var_cfg.range[0], var_cfg.range[1], var_cfg.n_bins + 1)


def load_parquet(path, n_max):
    data = ak.from_parquet(path)
    print(data.fields)
    if n_max > 0:
        data = data[:n_max]
    return data


if __name__ == "__main__":
    args = docopt(__doc__)

    signal_train = args["-i"]
    bkg_train = args["-b"]
    output_dir = args["-o"]
    n_files = int(args["-n"]) if args["-n"] else -1
    produce_plots = args["-p"]

    os.makedirs(output_dir, exist_ok=True)

    # ── load Hydra config for bin edges ───────────────────────────────────────
    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, job_name="compute_weights"):
        cfg = compose(config_name="weighting")

    wcfg = cfg.weighting
    p_edges = build_bin_edges(wcfg.variables.p)
    theta_edges = build_bin_edges(wcfg.variables.theta)

    # ── load data ─────────────────────────────────────────────────────────────
    sig_data = load_parquet(signal_train, n_files)
    bkg_data = load_parquet(bkg_train, n_files)

    # ── build normalised 2-D histograms (theta × pT) ──────────────────────────
    sig_matrix = wt.create_matrix(
        data=sig_data,
        y_bin_edges=theta_edges,
        x_bin_edges=p_edges,
        y_property="theta",
        x_property="p",
    )
    bkg_matrix = wt.create_matrix(
        data=bkg_data,
        y_bin_edges=theta_edges,
        x_bin_edges=p_edges,
        y_property="theta",
        x_property="p",
    )

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
    sig_out = os.path.join(output_dir, "sig_weights.npy")
    bkg_out = os.path.join(output_dir, "bkg_weights.npy")
    np.save(sig_out, sig_weight_matrix)
    np.save(bkg_out, bkg_weight_matrix)

    # Save bin edges alongside so apply_weights.py can reconstruct the lookup
    np.save(os.path.join(output_dir, "p_edges.npy"), p_edges)
    np.save(os.path.join(output_dir, "theta_edges.npy"), theta_edges)

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
