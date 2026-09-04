#!/usr/bin/env python3
"""Merge, split, compute weights, and apply weights — single-threaded, no subprocess.

Usage:
    python3 merge_and_weight.py --input-dir <dir> --out <dir>
                                [--short-signal z] [--short-bkg qq] [--train-frac 0.9]
                                [--chunk-files 100] [--seed 42]

Input dir should contain two subdirectories whose names contain "Z_tautau"
(signal) and "qq" (background), each with per-file ntupelized .parquet files.
All stages run sequentially in the main thread.

The merge/split stage is chunked and streamed so the full dataset is never held
in memory at once (see _merge_and_shuffle and _write_splits below).
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tqdm
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Allow importing weight_tools from the tools directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt

from ntupelizer.tools import general as g

# ---------------------------------------------------------------------------
# Merge + split (inlined from merge_files.py)
# ---------------------------------------------------------------------------

P4_COLUMNS = [
    "reco_jet_p4",
    "gen_jet_p4",
    "reco_cand_p4s",
    "gen_jet_tau_p4",
    "gen_jet_tau_full_p4",
    "event_reco_cand_p4s",
]


def _concat_tree(arrays, batch=100):
    """Concatenate a large list of arrays via a balanced tree (avoids recursion)."""
    if not arrays:
        return ak.Array([])
    level = arrays
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), batch):
            nxt.append(ak.concatenate(level[i : i + batch]))
        level = nxt
    return level[0]


def _normalize_p4(data: ak.Array) -> ak.Array:
    """Normalise p4 columns to the canonical {pt, eta, phi, energy} schema.

    reinitialize_p4 handles both {energy, x, y, z} (FastJet) and
    {pt, eta, phi, energy} representations, so mixing old and new files cannot
    produce a dense_union type that pyarrow refuses to write.

    The result is a plain (non-vector) record with explicit field names and
    non-nullable float64 fields.  vector.awk's internal spherical representation
    ({rho, phi, eta, t}) has a nullable `t` field, which makes Arrow schemas
    differ between chunks and breaks ParquetWriter.
    """
    p4_cols = [c for c in P4_COLUMNS if c in data.fields]
    if not p4_cols:
        return data
    out = {k: data[k] for k in data.fields if k not in p4_cols}
    for col in p4_cols:
        p4 = g.reinitialize_p4(data[col])
        out[col] = ak.zip(
            {
                "pt": ak.fill_none(p4.pt, 0.0),
                "eta": ak.fill_none(p4.eta, 0.0),
                "phi": ak.fill_none(p4.phi, 0.0),
                "energy": ak.fill_none(p4.energy, 0.0),
            }
        )
    return ak.Array(out)


def _merge_and_shuffle(files, chunk_files=100, seed=None):
    """Merge files into shuffled chunks on disk.

    Returns:
        chunk_paths: list of temporary chunk parquet paths (shuffled order)
        n_total:     total number of events across all chunks
        tmp_dir:     temporary directory holding the chunks (caller removes it)
    """
    if not files:
        raise FileNotFoundError("No .parquet files provided")

    rng = np.random.default_rng(seed) if seed is not None else np.random

    tmp_dir = tempfile.mkdtemp(prefix="merge_and_weight_chunks_")
    chunk_paths = []
    n_total = 0
    n_chunks = (len(files) + chunk_files - 1) // chunk_files

    for ci in range(n_chunks):
        chunk = files[ci * chunk_files : (ci + 1) * chunk_files]
        loaded = []
        for f in tqdm.tqdm(chunk, desc=f"Chunk {ci + 1}/{n_chunks}", leave=False):
            loaded.append(_normalize_p4(ak.from_parquet(f)))
        merged = _concat_tree(loaded)
        del loaded

        # Shuffle within the chunk.
        merged = merged[rng.permutation(len(merged))]

        chunk_path = os.path.join(tmp_dir, f"chunk_{ci:05d}.parquet")
        ak.to_parquet(
            merged,
            chunk_path,
            row_group_size=1024,
            compression="zstd",
            compression_level=6,
        )
        chunk_paths.append(chunk_path)
        n_total += len(merged)
        del merged

    # Shuffle the chunk order (coarse shuffle; within-chunk already shuffled).
    chunk_paths = [chunk_paths[i] for i in rng.permutation(len(chunk_paths))]

    print(
        f"Merged {n_total} events from {len(files)} files into {len(chunk_paths)} chunks"
    )
    return chunk_paths, n_total, tmp_dir


def _to_plain_table(arr):
    """Convert an awkward array to a metadata-free Arrow table.

    ak.to_arrow_table(extensionarray=False) still attaches awkward-specific
    schema/field metadata (option_type, ak:parameters, record_is_scalar) that
    can differ between chunks and make pyarrow's ParquetWriter reject a table
    whose field *types* are otherwise identical.  Stripping that metadata gives
    ParquetWriter a stable, comparable schema.
    """
    table = ak.to_arrow_table(arr, extensionarray=False)
    clean_schema = pa.schema(
        [field.with_metadata(None) for field in table.schema],
        metadata=None,
    )
    return table.cast(clean_schema)


def _write_splits(chunk_paths, n_total, output_dir, short_name, train_frac):
    """Stream shuffled chunks into train/test parquet files without holding all data."""
    n_train = int(n_total * train_frac)
    train_path = os.path.join(output_dir, f"{short_name}_train.parquet")
    test_path = os.path.join(output_dir, f"{short_name}_test.parquet")

    schema = _to_plain_table(ak.from_parquet(chunk_paths[0])).schema
    train_writer = pq.ParquetWriter(
        train_path,
        schema,
        compression="zstd",
        compression_level=6,
        use_byte_stream_split=True,
    )
    test_writer = pq.ParquetWriter(
        test_path,
        schema,
        compression="zstd",
        compression_level=6,
        use_byte_stream_split=True,
    )
    train_written = 0
    test_written = 0

    try:
        for cp in chunk_paths:
            arr = ak.from_parquet(cp)
            n = len(arr)
            n_to_train = min(n, max(0, n_train - train_written))
            if n_to_train > 0:
                train_writer.write_table(
                    _to_plain_table(arr[:n_to_train]), row_group_size=1024
                )
                train_written += n_to_train
            n_to_test = n - n_to_train
            if n_to_test > 0:
                test_writer.write_table(
                    _to_plain_table(arr[n_to_train:]), row_group_size=1024
                )
                test_written += n_to_test
            del arr
    finally:
        train_writer.close()
        test_writer.close()

    print(f"N={n_total}, Ntrain={train_written} Ntest={test_written}")
    return train_path, test_path


# ---------------------------------------------------------------------------
# Weights (inlined from compute_weights.py)
# ---------------------------------------------------------------------------


def build_bin_edges(var_cfg):
    return np.linspace(var_cfg.range[0], var_cfg.range[1], var_cfg.n_bins + 1)


def _gen_jet_theta_p(p4_struct):
    """Return (theta_deg, p) arrays from a gen_jet_p4 struct column/array."""
    pt = pc.struct_field(p4_struct, "pt").to_numpy()
    eta = pc.struct_field(p4_struct, "eta").to_numpy()
    theta = np.degrees(2.0 * np.arctan(np.exp(-eta)))
    p = pt * np.cosh(eta)
    return theta, p


def _accumulate_gen_jet_histogram(path, theta_edges, p_edges):
    """Accumulate the (theta, p) histogram of gen_jet_p4 by streaming batches.

    The weight matrix is only a 2D histogram, so we never need the whole train
    split in memory at once.  iter_batches reads a fixed number of rows at a time
    regardless of the file's row-group layout, keeping memory flat.
    """
    pf = pq.ParquetFile(path)
    hist = np.zeros((len(theta_edges) - 1, len(p_edges) - 1), dtype=np.float64)
    for batch in pf.iter_batches(batch_size=1024):
        theta, p = _gen_jet_theta_p(batch.column("gen_jet_p4"))
        h, _, _ = np.histogram2d(theta, p, bins=(theta_edges, p_edges))
        hist += h
    return hist


def compute_weight_matrices(sig_train_path, bkg_train_path, config_dir):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="weighting")
    wcfg = cfg.weighting
    p_edges = build_bin_edges(wcfg.variables.p)
    theta_edges = build_bin_edges(wcfg.variables.theta)

    sig_hist = _accumulate_gen_jet_histogram(sig_train_path, theta_edges, p_edges)
    bkg_hist = _accumulate_gen_jet_histogram(bkg_train_path, theta_edges, p_edges)

    sig_matrix = sig_hist / np.sum(sig_hist)
    bkg_matrix = bkg_hist / np.sum(bkg_hist)

    sig_w = wt.get_weight_matrix(target_matrix=sig_matrix, comp_matrix=bkg_matrix)
    bkg_w = wt.get_weight_matrix(target_matrix=bkg_matrix, comp_matrix=sig_matrix)

    return sig_w, bkg_w, p_edges, theta_edges


# ---------------------------------------------------------------------------
# Apply weights (inlined from apply_weights.py)
# ---------------------------------------------------------------------------


def _compute_weights_from_struct(p4_struct, weight_matrix, theta_edges, p_edges):
    """Compute per-jet weights from a gen_jet_p4 struct column (pyarrow-native)."""
    theta, p = _gen_jet_theta_p(p4_struct)
    n_theta, n_p = weight_matrix.shape
    theta_centers = (theta_edges[1:] + theta_edges[:-1]) / 2
    p_centers = (p_edges[1:] + p_edges[:-1]) / 2
    theta_bin = np.clip(np.digitize(theta, theta_centers) - 1, 0, n_theta - 1)
    p_bin = np.clip(np.digitize(p, p_centers) - 1, 0, n_p - 1)
    return weight_matrix[theta_bin, p_bin]


def apply_weights_and_save(
    input_path, weight_matrix, theta_edges, pt_edges, output_path
):
    """Stream-apply weights using pyarrow batches (stable schema, bounded memory)."""
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
    n_total = 0
    try:
        for batch in pf.iter_batches(batch_size=1024):
            table = pa.Table.from_batches([batch])
            weights = _compute_weights_from_struct(
                table.column("gen_jet_p4"), weight_matrix, theta_edges, pt_edges
            )
            table = table.append_column("cls_weight", pa.array(weights))
            writer.write_table(table, row_group_size=1024)
            n_total += table.num_rows
            del table
    finally:
        writer.close()

    print(f"Wrote {n_total} events → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    sys.setrecursionlimit(50000)  # awkward concatenation can recurse deeply
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        required=True,
        help="Parent dir with p8_ee_Z_tautau_ecm91/ and p8_ee_Z_qq_ecm91/ subdirs",
    )
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--short-signal", default="z")
    p.add_argument("--short-bkg", default="qq")
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument(
        "--chunk-files",
        type=int,
        default=100,
        help="Number of input files to merge per in-memory chunk",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for shuffling")
    args = p.parse_args()

    out_dir = args.out
    short_sig = args.short_signal
    short_bkg = args.short_bkg
    os.makedirs(out_dir, exist_ok=True)

    # Discover signal/background — subdirs, or flat files, or input-dir itself
    subdirs = [
        d
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ]

    if subdirs:
        sig_sub = next((d for d in subdirs if "tautau" in d), None)
        bkg_sub = next((d for d in subdirs if "qq" in d and "tautau" not in d), None)
        if sig_sub and bkg_sub:
            sig_files = sorted(
                glob.glob(os.path.join(args.input_dir, sig_sub, "*.parquet"))
            )
            bkg_files = sorted(
                glob.glob(os.path.join(args.input_dir, bkg_sub, "*.parquet"))
            )
    if not subdirs or not (sig_sub and bkg_sub):
        # Flat-file mode: parquets directly in input-dir, or input-dir IS a dataset
        if os.path.isdir(args.input_dir):
            flat = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
        else:
            flat = []
        sig_files = [f for f in flat if "tautau" in os.path.basename(f)]
        bkg_files = [
            f
            for f in flat
            if "qq" in os.path.basename(f) and "tautau" not in os.path.basename(f)
        ]

    if not sig_files or not bkg_files:
        print(f"ERROR: could not find signal/background files in {args.input_dir}")
        print(f"  Signal candidates: {len(sig_files)}")
        print(f"  Bkg    candidates: {len(bkg_files)}")
        sys.exit(1)

    print(f"Signal:      {len(sig_files)} files")
    print(f"Background:  {len(bkg_files)} files")

    # Config directory for weighting (relative to the tools dir)
    config_dir = str(Path(__file__).resolve().parents[1] / "config")

    # ── Stage 1: merge + split (per dataset, saved immediately) ─────────
    print("\n=== Stage 1: Merge & split ===")
    split_dir = os.path.join(out_dir, "split")
    os.makedirs(split_dir, exist_ok=True)

    for files, name in [(sig_files, short_sig), (bkg_files, short_bkg)]:
        train_path = os.path.join(split_dir, f"{name}_train.parquet")
        test_path = os.path.join(split_dir, f"{name}_test.parquet")
        # Also check output dir directly (legacy location)
        train_alt = os.path.join(out_dir, f"{name}_train.parquet")
        test_alt = os.path.join(out_dir, f"{name}_test.parquet")
        if (os.path.exists(train_path) and os.path.exists(test_path)) or (
            os.path.exists(train_alt) and os.path.exists(test_alt)
        ):
            print(f"\n--- {name} (already exists, skipping) ---")
            continue
        print(f"\n--- {name} ---")
        chunk_paths, n_total, tmp_dir = _merge_and_shuffle(
            files, chunk_files=args.chunk_files, seed=args.seed
        )
        try:
            _write_splits(chunk_paths, n_total, split_dir, name, args.train_frac)
        finally:
            shutil.rmtree(tmp_dir)
        print(f"Saved {name}_train.parquet, {name}_test.parquet")

    # ── Stage 2: compute weights ────────────────────────────────────────
    print("\n=== Stage 2: Compute weights ===")

    def _find_split(name, split):
        a = os.path.join(split_dir, f"{name}_{split}.parquet")
        b = os.path.join(out_dir, f"{name}_{split}.parquet")
        return a if os.path.exists(a) else b

    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    sig_w_path = os.path.join(weights_dir, "sig_weights.npy")
    bkg_w_path = os.path.join(weights_dir, "bkg_weights.npy")
    p_edges_path = os.path.join(weights_dir, "p_edges.npy")
    theta_edges_path = os.path.join(weights_dir, "theta_edges.npy")

    if (
        os.path.exists(sig_w_path)
        and os.path.exists(bkg_w_path)
        and os.path.exists(p_edges_path)
        and os.path.exists(theta_edges_path)
    ):
        sig_w = np.load(sig_w_path)
        bkg_w = np.load(bkg_w_path)
        p_edges = np.load(p_edges_path)
        theta_edges = np.load(theta_edges_path)
        print(f"Weight matrices already exist, loading from {weights_dir}/")
    else:
        sig_train_path = _find_split(short_sig, "train")
        bkg_train_path = _find_split(short_bkg, "train")
        sig_w, bkg_w, p_edges, theta_edges = compute_weight_matrices(
            sig_train_path, bkg_train_path, config_dir
        )
        np.save(sig_w_path, sig_w)
        np.save(bkg_w_path, bkg_w)
        np.save(p_edges_path, p_edges)
        np.save(theta_edges_path, theta_edges)
        print(f"Saved weight matrices to {weights_dir}/")

    # ── Stage 3: apply weights ──────────────────────────────────────────
    print("\n=== Stage 3: Apply weights ===")
    for split_name in ["train", "test"]:
        s_path = _find_split(short_sig, split_name)
        b_path = _find_split(short_bkg, split_name)
        apply_weights_and_save(
            s_path,
            sig_w,
            theta_edges,
            p_edges,
            os.path.join(out_dir, f"{short_sig}_{split_name}.parquet"),
        )
        apply_weights_and_save(
            b_path,
            bkg_w,
            theta_edges,
            p_edges,
            os.path.join(out_dir, f"{short_bkg}_{split_name}.parquet"),
        )

    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()
