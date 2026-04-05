"""Apply pre-computed weight matrix to a split parquet file.

Reads the weight matrix and bin edges produced by compute_weights.py from
--weights-dir, looks up the per-event weight for each jet based on its
(theta, pT) bin, and writes the input file back out with an additional
'weight' column.

The correct matrix (sig or bkg) is selected via --is-signal.

Usage:
    apply_weights.py -i <input> -w <weights_dir> -o <output> --is-signal <bool>

Options:
    -i <input>          Path to the input parquet file (e.g. z_train.parquet).
    -w <weights_dir>    Directory containing sig_weights.npy, bkg_weights.npy,
                        pt_edges.npy and theta_edges.npy (produced by
                        compute_weights.py).
    -o <output>         Path for the output parquet file with the 'weight' column
                        added.
    --is-signal <bool>  'true' if this file is from the signal dataset, 'false'
                        for background.  Selects sig_weights.npy vs bkg_weights.npy.
"""

import os
import sys
import numpy as np
import awkward as ak
from pathlib import Path
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt


if __name__ == "__main__":
    args = docopt(__doc__)

    input_path = args["-i"]
    weights_dir = args["-w"]
    output_path = args["-o"]
    is_signal = args["--is-signal"].lower() in ("true", "1", "yes")

    # ── load weight matrix and bin edges ──────────────────────────────────────
    matrix_file = "sig_weights.npy" if is_signal else "bkg_weights.npy"
    weight_matrix = np.load(os.path.join(weights_dir, matrix_file))
    pt_edges = np.load(os.path.join(weights_dir, "pt_edges.npy"))
    theta_edges = np.load(os.path.join(weights_dir, "theta_edges.npy"))

    # ── load data and compute per-event weights ────────────────────────────────
    data = ak.from_parquet(input_path)
    weights = wt.get_weights(data, weight_matrix, theta_edges, pt_edges)

    # ── write output with the new 'weight' column ──────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = ak.Array(
        {**{field: data[field] for field in data.fields}, "cls_weight": weights}
    )
    ak.to_parquet(out, output_path, row_group_size=1024)

    print(f"Wrote {len(data)} events with weights → {output_path}")
