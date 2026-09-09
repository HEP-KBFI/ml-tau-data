"""Apply pre-computed weight matrices to signal and background parquet files.

Standalone utility, not part of the Snakemake workflow: there the weights are
applied by merge_files.py in the same pass that writes the output chunks. Use
this to re-weight an already written dataset without re-merging it.

Reads the weight matrices and bin edges produced by compute_weights.py from
--weights-dir, looks up the per-event weight for each jet based on its
(theta, p) bin, and writes weighted output parquet files to --output-dir.

The weight is applied by streaming the input via pyarrow row groups, so the
full split is never held in memory and the schema stays stable.

Inputs may be chunked: pointing -i/-b at a directory of
<short>_<split>_<index>.parquet files weights every chunk of that split and
writes one output file per input file, keeping the chunk filenames unchanged.

Usage:
    apply_weights.py -i <signal> -b <background> -w <weights_dir> -o <output_dir>
                     [-t <split>] [--row-group-size <n>] [-p]

Options:
    -i <signal>         Signal input: a single parquet file, or a directory of
                        chunked files (e.g. z_train_00000.parquet, z_train_00001.parquet).
    -b <background>     Background input, same two forms as -i.
    -w <weights_dir>    Directory containing sig_weights.npy, bkg_weights.npy,
                        pt_edges.npy and theta_edges.npy (produced by
                        compute_weights.py).
    -o <output_dir>     Directory for the output parquet files. Output filenames
                        match the input filenames.
    -t <split>          Split to select when -i/-b are directories.
                        [default: train]
    --row-group-size <n>  Rows per parquet row group, and per streamed batch.
                        [default: 1024]
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
import general as g
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


def apply_and_save(
    input_path, weight_matrix, theta_edges, pt_edges, output_dir, row_group_size=1024
):
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
        # One row group in, one row group out: the batch size doubles as the
        # output row group size, so the input row group layout is not inherited.
        for batch in pf.iter_batches(batch_size=row_group_size):
            table = pa.Table.from_batches([batch])
            weights = _compute_weights_from_struct(
                table.column("gen_jet_p4"), weight_matrix, theta_edges, pt_edges
            )
            table = table.append_column("cls_weight", pa.array(weights))
            writer.write_table(table, row_group_size=row_group_size)
            all_weights.append(weights)
            n_total += table.num_rows
            del table
        if n_total == 0:
            # An empty input still has to get an explicit empty row group:
            # a parquet file with zero row groups cannot be read by awkward.
            writer.write_table(out_schema.empty_table())
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
    split = args["-t"] or "train"
    row_group_size = int(args["--row-group-size"] or 1024)
    produce_plots = args["-p"]

    # ── load weight matrices and bin edges ────────────────────────────────────
    sig_weight_matrix = np.load(os.path.join(weights_dir, "sig_weights.npy"))
    bkg_weight_matrix = np.load(os.path.join(weights_dir, "bkg_weights.npy"))
    pt_edges = np.load(os.path.join(weights_dir, "p_edges.npy"))
    theta_edges = np.load(os.path.join(weights_dir, "theta_edges.npy"))

    os.makedirs(output_dir, exist_ok=True)

    # ── resolve chunked inputs (directory) or a single file ───────────────────
    sig_paths = g.split_chunk_paths(sig_path, split)
    bkg_paths = g.split_chunk_paths(bkg_path, split)

    # ── apply weights and save (streamed, one output file per input file) ─────
    def _apply_all(paths, weight_matrix):
        # Only the per-jet weights are kept in memory across chunks (one float
        # per jet), which the optional plots below need; the jets themselves are
        # streamed straight to disk.
        per_file = [
            apply_and_save(
                p, weight_matrix, theta_edges, pt_edges, output_dir, row_group_size
            )
            for p in paths
        ]
        return np.concatenate(per_file) if per_file else np.array([])

    sig_weights = _apply_all(sig_paths, sig_weight_matrix)
    bkg_weights = _apply_all(bkg_paths, bkg_weight_matrix)

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
        # A single chunk (O(100k) jets) is plenty for a shape comparison, so
        # only the first one is read instead of the whole split.
        sig_data = ak.from_parquet(sig_paths[0], columns=list(error_vars.keys()))
        bkg_data = ak.from_parquet(bkg_paths[0], columns=list(error_vars.keys()))
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
