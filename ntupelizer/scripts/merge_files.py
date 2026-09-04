"""Merge per-file parquets for one sample, shuffle, and split into train/test.

Usage:
    merge_files.py -i <input_dir> -o <output_dir> -s <sample_shortname> -f <train_frac>
                   [--chunk-files <n>] [--seed <n>]

Options:
    -i <input_dir>         Directory containing the per-file .parquet inputs.
    -o <output_dir>        Directory where <sample_shortname>_train.parquet and
                           <sample_shortname>_test.parquet will be written.
    -s <sample_shortname>  Short identifier for the sample (e.g. "qq", "zh", "z").
    -f <train_frac>        Fraction of events to assign to the train split (e.g. 0.7).
    --chunk-files <n>      Number of input files to merge per in-memory chunk.
                           Lower values use less memory but create more chunks.
                           [default: 100]
    --seed <n>             Random seed for reproducible shuffling. [default: None]

This script merges and shuffles in chunks so that the full dataset is never held
in memory at once.  Each chunk is loaded, normalised, tree-concatenated, shuffled,
and written to a temporary parquet.  The chunks are then shuffled and streamed
into the final train/test files with a ParquetWriter.
"""

import glob
import os
import shutil
import sys
import tempfile

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from docopt import docopt

from ntupelizer.tools import general as g

P4_COLUMNS = [
    "reco_jet_p4",
    "gen_jet_p4",
    "reco_cand_p4s",
    "gen_jet_tau_p4",
    "gen_jet_tau_full_p4",
    "event_reco_cand_p4s",
]


def _concat_tree(arrays, batch=100):
    """Concatenate a large list of arrays via a balanced tree.

    This keeps peak memory lower than a single ak.concatenate(list) because
    intermediate results are produced and released in batches.
    """
    if not arrays:
        return ak.Array([])
    level = arrays
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), batch):
            nxt.append(ak.concatenate(level[i : i + batch]))
        level = nxt
    return level[0]


def _normalize_p4(data):
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


def _discover_columns(files, wanted_columns):
    """Return the subset of wanted_columns present in the input files."""
    first = ak.from_parquet(files[0])
    available = first.fields
    columns = [c for c in wanted_columns if c in available]
    missing = [c for c in wanted_columns if c not in available]
    if missing:
        print(f"Columns not in file (skipping): {missing}")
    return columns


def load_sample(path, chunk_files=100, seed=None):
    """Merge files into shuffled chunks on disk.

    Returns:
        chunk_paths: list of temporary chunk parquet paths (shuffled order)
        n_total:     total number of events across all chunks
        tmp_dir:     temporary directory holding the chunks (caller removes it)
    """
    wanted_columns = [
        # basic reco inputs
        "reco_jet_p4",
        "reco_cand_p4s",
        "reco_cand_charges",
        "reco_cand_pdgs",
        # advanced reco inputs: track impact parameters
        "reco_cand_dz",
        "reco_cand_dz_error",
        "reco_cand_dxy",
        "reco_cand_dxy_error",
        # targets
        "gen_jet_p4",  # generated jet p4
        "gen_jet_tau_p4",  # tau visible momentum, excluding neutrino
        "gen_jet_tau_decaymode",  # tau decay mode
        "gen_jet_tau_charge",
        # tau daughter info (only present with DecayProductNtupelizer)
        "gen_jet_tau_vis_daughter_p4s",
        "gen_jet_tau_vis_daughter_pdgs",
        "gen_jet_tau_vis_daughter_charges",
    ]

    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {path}")

    columns = _discover_columns(files, wanted_columns)

    rng = np.random.default_rng(seed) if seed is not None else np.random

    tmp_dir = tempfile.mkdtemp(prefix="merge_files_")
    chunk_paths = []
    n_total = 0
    n_chunks = (len(files) + chunk_files - 1) // chunk_files

    for ci in range(n_chunks):
        chunk = files[ci * chunk_files : (ci + 1) * chunk_files]
        loaded = []
        for f in tqdm.tqdm(chunk, desc=f"Chunk {ci + 1}/{n_chunks}", leave=False):
            loaded.append(_normalize_p4(ak.from_parquet(f, columns=columns)))
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


def split_and_write(chunk_paths, n_total, output_dir, short_name, train_frac):
    """Stream chunks into train/test parquet files without holding all data."""
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


if __name__ == "__main__":
    sys.setrecursionlimit(50000)  # awkward concatenation can recurse deeply
    args = docopt(__doc__)

    sample_shortname = args["-s"]
    input_dir = args["-i"]
    train_frac = float(args["-f"])
    output_dir = args["-o"]
    chunk_files = int(args["--chunk-files"] or 100)
    seed = None if args["--seed"] in (None, "None") else int(args["--seed"])

    os.makedirs(output_dir, exist_ok=True)

    chunk_paths, n_total, tmp_dir = load_sample(
        input_dir, chunk_files=chunk_files, seed=seed
    )
    try:
        split_and_write(chunk_paths, n_total, output_dir, sample_shortname, train_frac)
    finally:
        shutil.rmtree(tmp_dir)
