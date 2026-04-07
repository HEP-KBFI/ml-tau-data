"""Apply pre-computed weight matrices to signal and background parquet files.

Reads the weight matrices and bin edges produced by compute_weights.py from
--weights-dir, looks up the per-event weight for each jet based on its
(theta, p) bin, and writes weighted output parquet files to --output-dir.

Usage:
    apply_weights.py -i <signal> -b <background> -w <weights_dir> -o <output_dir> [-p]

Options:
    -i <signal>         Path to the signal parquet file (e.g. z_train.parquet).
    -b <background>     Path to the background parquet file (e.g. qq_train.parquet).
    -w <weights_dir>    Directory containing sig_weights.npy, bkg_weights.npy,
                        pt_edges.npy and theta_edges.npy (produced by
                        compute_weights.py).
    -o <output_dir>     Directory for the output parquet files. Output filenames
                        match the input filenames.
    -p                  Produce a weight distribution plot. [default: False]
"""

import os
import sys
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt


def apply_and_save(input_path, weight_matrix, theta_edges, pt_edges, output_dir):
    data = ak.from_parquet(input_path)
    weights = wt.get_weights(data, weight_matrix, theta_edges, pt_edges)
    output_path = os.path.join(output_dir, Path(input_path).name)
    out = ak.Array(
        {**{field: data[field] for field in data.fields}, "cls_weight": weights}
    )
    ak.to_parquet(out, output_path, row_group_size=1024)
    print(f"Wrote {len(data)} events with weights → {output_path}")
    return weights, data


if __name__ == "__main__":
    args = docopt(__doc__)

    sig_path = args["-i"]
    bkg_path = args["-b"]
    weights_dir = args["-w"]
    output_dir = args["-o"]
    produce_plots = args["-p"]

    # ── load weight matrices and bin edges ────────────────────────────────────
    sig_weight_matrix = np.load(os.path.join(weights_dir, "sig_weights.npy"))
    bkg_weight_matrix = np.load(os.path.join(weights_dir, "bkg_weights.npy"))
    pt_edges = np.load(os.path.join(weights_dir, "p_edges.npy"))
    theta_edges = np.load(os.path.join(weights_dir, "theta_edges.npy"))

    os.makedirs(output_dir, exist_ok=True)

    # ── apply weights and save ────────────────────────────────────────────────
    sig_weights, sig_data = apply_and_save(
        sig_path, sig_weight_matrix, theta_edges, pt_edges, output_dir
    )
    bkg_weights, bkg_data = apply_and_save(
        bkg_path, bkg_weight_matrix, theta_edges, pt_edges, output_dir
    )

    # ── optional weight distribution plot ─────────────────────────────────────
    if produce_plots:
        validation_dir = os.path.join(output_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        wt.plot_weight_distributions(sig_weights, bkg_weights, validation_dir)
        print("Saved weight distribution plot.")

        # ── dxy / dz error overlay plots ──────────────────────────────────────
        INVALID = -1000.0
        error_vars = {
            "reco_cand_dxy_error": "PFCandidate dxy error [mm]",
            "reco_cand_dz_error": "PFCandidate dz error [mm]",
        }
        log_bins = np.logspace(-4, 0, 80)
        for var, xlabel in error_vars.items():
            if var not in sig_data.fields:
                continue
            fig, ax = plt.subplots(figsize=(7, 5.5))
            for data, label, color in [
                (sig_data, "Signal", "red"),
                (bkg_data, "Background", "blue"),
            ]:
                flat = ak.to_numpy(ak.flatten(data[var]))
                flat = flat[flat > INVALID + 1]
                counts, edges = np.histogram(flat, bins=log_bins, density=True)
                ax.step(edges[:-1], counts, where="post", label=label, color=color)
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Fraction [a.u.]")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(validation_dir, f"{var}.pdf"), bbox_inches="tight")
            plt.close(fig)
        print("Saved dxy/dz error plots.")
