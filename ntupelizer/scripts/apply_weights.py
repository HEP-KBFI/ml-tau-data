"""Apply pre-computed weight matrices to signal and background parquet files.

Reads the weight matrices and bin edges produced by compute_weights.py from
--weights-dir, looks up the per-event weight for each jet based on its
(theta, p) bin, and writes weighted output parquet files to --output-dir.

The weight is applied by streaming the input via pyarrow row groups, so the
full split is never held in memory and the schema stays stable.

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
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt


def _gen_jet_theta_p(p4_struct):
    """Return (theta_deg, p) arrays from a gen_jet_p4 struct column/array."""
    pt = pc.struct_field(p4_struct, "pt").to_numpy()
    eta = pc.struct_field(p4_struct, "eta").to_numpy()
    theta = np.degrees(2.0 * np.arctan(np.exp(-eta)))
    p = pt * np.cosh(eta)
    return theta, p


def _compute_weights_from_struct(p4_struct, weight_matrix, theta_edges, p_edges):
    """Compute per-jet weights from a gen_jet_p4 struct column (pyarrow-native)."""
    theta, p = _gen_jet_theta_p(p4_struct)
    n_theta, n_p = weight_matrix.shape
    theta_centers = (theta_edges[1:] + theta_edges[:-1]) / 2
    p_centers = (p_edges[1:] + p_edges[:-1]) / 2
    theta_bin = np.clip(np.digitize(theta, theta_centers) - 1, 0, n_theta - 1)
    p_bin = np.clip(np.digitize(p, p_centers) - 1, 0, n_p - 1)
    return weight_matrix[theta_bin, p_bin]


def apply_and_save(input_path, weight_matrix, theta_edges, pt_edges, output_dir):
    output_path = os.path.join(output_dir, Path(input_path).name)

    pf = pq.ParquetFile(input_path)
    schema = pa.schema([f.with_metadata(None) for f in pf.schema_arrow], metadata=None)
    out_schema = schema.append(pa.field("cls_weight", pa.float64()))

    writer = pq.ParquetWriter(
        output_path,
        out_schema,
        compression="zstd",
        compression_level=6,
        use_byte_stream_split=True,
    )
    all_weights = []
    n_total = 0
    try:
        for batch in pf.iter_batches(batch_size=1024):
            table = pa.Table.from_batches([batch])
            weights = _compute_weights_from_struct(
                table.column("gen_jet_p4"), weight_matrix, theta_edges, pt_edges
            )
            table = table.append_column("cls_weight", pa.array(weights))
            writer.write_table(table, row_group_size=1024)
            all_weights.append(weights)
            n_total += table.num_rows
            del table
    finally:
        writer.close()

    all_weights = np.concatenate(all_weights) if all_weights else np.array([])
    print(f"Wrote {n_total} events with weights → {output_path}")
    return all_weights


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

    # ── apply weights and save (streamed) ─────────────────────────────────────
    sig_weights = apply_and_save(
        sig_path, sig_weight_matrix, theta_edges, pt_edges, output_dir
    )
    bkg_weights = apply_and_save(
        bkg_path, bkg_weight_matrix, theta_edges, pt_edges, output_dir
    )

    # ── optional weight distribution plot ─────────────────────────────────────
    if produce_plots:
        validation_dir = os.path.join(output_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        wt.plot_weight_distributions(sig_weights, bkg_weights, validation_dir)
        print("Saved weight distribution plot.")

        # ── dxy / dz error overlay plots (only load the two needed columns) ──
        INVALID = -1000.0
        error_vars = {
            "reco_cand_dxy_error": "PFCandidate dxy error [mm]",
            "reco_cand_dz_error": "PFCandidate dz error [mm]",
        }
        log_bins = np.logspace(-4, 0, 80)
        sig_data = ak.from_parquet(sig_path, columns=list(error_vars.keys()))
        bkg_data = ak.from_parquet(bkg_path, columns=list(error_vars.keys()))
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
