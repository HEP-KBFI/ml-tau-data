"""Fill train/test parquet chunks from the ntupelized files of one sample.

Usage:
    merge_files.py -i <input_dir> -o <output_dir> -s <sample_shortname> -f <train_frac>
                   [-w <weights_dir> --side <side>]
                   [--chunk-size <n>] [--row-group-size <n>] [--seed <n>]

Options:
    -i <input_dir>         Directory containing the per-file .parquet inputs.
    -o <output_dir>        Directory where <sample_shortname>_<split>_<index>.parquet
                           files will be written (e.g. z_train_00000.parquet).
    -s <sample_shortname>  Short identifier for the sample (e.g. "qq", "zh", "z").
    -f <train_frac>        Fraction of events to assign to the train split (e.g. 0.7).
    -w <weights_dir>       Directory of weight matrices and bin edges from
                           compute_weights.py. When given, a cls_weight column is
                           added to the output as it is written; without it the
                           output is unweighted.
    --side <side>          Which matrix of <weights_dir> to apply, "sig" or "bkg".
                           Required together with -w.
    --chunk-size <n>       Number of events (rows) per output .parquet file.
                           Each split is written as a numbered series of files of
                           exactly this many rows, apart from the last file of
                           the split, which holds the remainder. [default: 100000]
    --row-group-size <n>   Rows per parquet row group. Smaller groups make
                           partial reads cheaper and full reads slower.
                           [default: 1024]
    --seed <n>             Random seed for the train/test assignment and the
                           in-buffer shuffle. [default: None]

The inputs are consumed one row group at a time: each row group is normalised,
weighted, split between train and test, and appended to that split's fill
buffer.  A buffer writes an output file as soon as it holds --chunk-size events,
and when an input file runs out the next one is opened and keeps filling the
file that is still open.  So every event is read once and written once — there
is no intermediate merged copy of the sample, and memory is bounded by one
output file rather than by the size of the sample.

The cost of streaming is the reach of the shuffle: events are mixed within a
fill buffer (one output file), not across the whole sample.  Inputs are
independent simulation runs of the same process, so this only matters if the
training loader relies on the file order itself being random.
"""

import glob
import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from docopt import docopt

from ntupelizer.tools import general as g

# Allow importing weight_tools from the tools directory (it imports its
# siblings by bare module name), matching compute_weights.py / apply_weights.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt

# Rows per parquet row group.  Small row groups keep partial reads cheap, which
# is what the training dataloader wants, at the cost of full-file reads: measured
# on a 100k-jet file, 1024 rows per group (98 groups) reads in full 5.4x slower
# than 20000 rows per group (5 groups), is ~7% bigger on disk and has a 375 KB
# rather than 25 KB footer.  Raise this if bulk reads matter more than
# granularity for a given dataset.
DEFAULT_ROW_GROUP_SIZE = 1024

# Width of the zero-padded chunk index in output filenames
# (z_train_00000.parquet).  Padding keeps a plain lexicographic listing of a
# split in the same order as its chunk sequence.
CHUNK_INDEX_WIDTH = 5

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


WANTED_COLUMNS = [
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


def input_files(path):
    """Return the ntupelized .parquet inputs of one sample, in a stable order."""
    files = sorted(glob.glob(os.path.join(path, "*.parquet")), key=g.natural_key)
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {path}")
    return files


def count_events(files):
    """Total number of jets across the inputs, from parquet footers only.

    Reading the footers costs a seek per file and no data, and knowing the total
    up front is what lets the train/test split come out at exactly the requested
    fraction while the events are only ever streamed once (see _draw_test_mask).
    """
    n_total = 0
    for path in tqdm.tqdm(files, desc="Counting", leave=False):
        n_total += ak.metadata_from_parquet(path)["num_rows"]
    return n_total


def iter_row_groups(files, columns, weighter=None):
    """Yield the row groups of the inputs one at a time, ready to be written.

    One row group is the unit of reading: it is normalised, weighted, and handed
    over, so the memory held at any moment is one row group plus whatever the
    caller is buffering — never a whole input file, let alone the whole sample.
    Inputs are opened in order and each is exhausted before the next is opened.
    """
    for path in tqdm.tqdm(files, desc="Filling", unit="file"):
        metadata = ak.metadata_from_parquet(path)
        if metadata["num_rows"] == 0:
            print(f"WARNING: {path} holds no jets, skipping")
            continue
        for rg in range(metadata["num_row_groups"]):
            data = _normalize_p4(
                ak.from_parquet(path, row_groups=[rg], columns=columns)
            )
            if weighter is not None:
                data = ak.with_field(data, weighter(data), "cls_weight")
            yield data


def _draw_test_mask(n_rows, n_test_left, n_left, rng):
    """Pick which of the next n_rows events belong to the test split.

    Drawing the count from a hypergeometric distribution and then choosing that
    many positions at random makes the streamed split identical to taking a
    uniformly random subset of n_test events out of the whole sample: the test
    events come from all inputs rather than from a tail of them, and the split
    sizes still land exactly on the requested fraction.
    """
    n_take = int(
        rng.hypergeometric(
            ngood=n_test_left, nbad=n_left - n_test_left, nsample=n_rows
        )
    )
    mask = np.zeros(n_rows, dtype=bool)
    if n_take:
        mask[rng.choice(n_rows, size=n_take, replace=False)] = True
    return mask


class FillBuffer:
    """Accumulate streamed row groups and flush them to a writer in blocks.

    The buffer exists so that the events of an output file can be shuffled
    before they are written: they arrive in input order, and consecutive jets
    come from the same event. It holds at most `capacity` events plus the last
    row group appended, so its memory footprint is bounded by the size of one
    output file.
    """

    def __init__(self, writer, capacity, rng):
        self.writer = writer
        self.capacity = int(capacity)
        self.rng = rng
        self._parts = []
        self._n_buffered = 0

    def append(self, data):
        if len(data) == 0:
            return
        self._parts.append(data)
        self._n_buffered += len(data)
        while self._n_buffered >= self.capacity:
            self.flush(block=self.capacity)

    def flush(self, block=None):
        """Shuffle what is buffered and hand `block` rows to the writer.

        Handing over whole blocks of `capacity` rows, and keeping the rest
        buffered, means the writer fills each output file in a single call and
        its row groups all come out the configured size — a flush of a partial
        block would end up as a short row group in the middle of a file.
        Without `block`, everything left is handed over (end of the stream).
        """
        if not self._parts:
            return
        merged = _concat_tree(self._parts)
        merged = merged[self.rng.permutation(len(merged))]
        if block is not None and len(merged) > block:
            head, tail = merged[:block], merged[block:]
        else:
            head, tail = merged, None
        self._parts = [] if tail is None else [tail]
        self._n_buffered = 0 if tail is None else len(tail)
        self.writer.write(_to_plain_table(head))


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


def load_weighter(weights_dir, side):
    """Return a callable adding per-jet weights, or None if no weights are given.

    The returned function takes a merged ntuple array and returns the cls_weight
    column for it, looked up in the (theta, p) matrix of `side` ("sig"/"bkg")
    that compute_weights.py wrote to `weights_dir`.
    """
    if not weights_dir:
        return None
    if side not in ("sig", "bkg"):
        raise ValueError(f"--side must be 'sig' or 'bkg' when -w is given, got {side!r}")
    weight_matrix = np.load(os.path.join(weights_dir, f"{side}_weights.npy"))
    p_edges = np.load(os.path.join(weights_dir, "p_edges.npy"))
    theta_edges = np.load(os.path.join(weights_dir, "theta_edges.npy"))
    print(f"Applying {side} weights from {weights_dir}")

    def weigh(data):
        return wt.get_weights(data, weight_matrix, theta_edges, p_edges)

    return weigh


class ChunkedParquetWriter:
    """Write a stream of Arrow tables into numbered parquet files of fixed size.

    Output files are named <short_name>_<split>_<index>.parquet with a
    zero-based index padded to CHUNK_INDEX_WIDTH digits, and hold at most
    `chunk_size` rows each (the final file of a split is usually shorter).
    Incoming tables are sliced at the chunk boundary, so the row count per file
    does not depend on how the caller batches its writes.
    """

    def __init__(
        self,
        output_dir,
        short_name,
        split,
        schema,
        chunk_size,
        row_group_size=DEFAULT_ROW_GROUP_SIZE,
    ):
        self.output_dir = output_dir
        self.short_name = short_name
        self.split = split
        self.schema = schema
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        # A row group can never usefully be larger than the file holding it.
        self.row_group_size = min(int(row_group_size), self.chunk_size)
        if self.row_group_size < 1:
            raise ValueError(f"row_group_size must be >= 1, got {row_group_size}")
        self.paths = []
        self.total_rows = 0
        self._writer = None
        self._index = 0
        self._rows_in_chunk = 0

    def _path(self, index):
        return os.path.join(
            self.output_dir,
            f"{self.short_name}_{self.split}_{index:0{CHUNK_INDEX_WIDTH}d}.parquet",
        )

    def _open(self):
        path = self._path(self._index)
        self._writer = pq.ParquetWriter(
            path,
            self.schema,
            compression="zstd",
            compression_level=6,
            use_byte_stream_split=True,
        )
        self.paths.append(path)
        self._rows_in_chunk = 0

    def write(self, table):
        """Append a table, rolling over to a new file at every chunk boundary."""
        offset = 0
        n_rows = table.num_rows
        while offset < n_rows:
            if self._writer is None:
                self._open()
            take = min(self.chunk_size - self._rows_in_chunk, n_rows - offset)
            self._writer.write_table(
                table.slice(offset, take), row_group_size=self.row_group_size
            )
            offset += take
            self._rows_in_chunk += take
            self.total_rows += take
            if self._rows_in_chunk >= self.chunk_size:
                self.close()
                self._index += 1

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def finalize(self):
        """Close the open file and guarantee the split has at least one file.

        Downstream stages address a split by its first chunk (e.g. the
        validation plots read z_train_00000.parquet), so an empty split still gets
        a <short_name>_<split>_0.parquet with the correct schema rather than no
        file at all.  The empty table has to be written explicitly: a parquet
        file with zero row groups cannot be read back by awkward.
        """
        self.close()
        if not self.paths:
            self._open()
            self._writer.write_table(self.schema.empty_table())
            self.close()
        return self.paths


def fill_splits(
    input_dir,
    output_dir,
    short_name,
    train_frac,
    chunk_size=100_000,
    row_group_size=DEFAULT_ROW_GROUP_SIZE,
    weighter=None,
    seed=None,
):
    """Stream the inputs of one sample into weighted train/test chunk files.

    Every event is read once and written once: row groups are pulled from the
    inputs in turn, each event is assigned to the train or the test split, and
    the two fill buffers write out an output file whenever they have gathered
    chunk_size events.  When an input is exhausted the next one is opened and
    keeps filling the file that is still open, so the outputs are all exactly
    chunk_size events apart from the last one of each split.
    """
    files = input_files(input_dir)
    columns = _discover_columns(files, WANTED_COLUMNS)
    n_total = count_events(files)
    n_test_left = n_total - int(n_total * train_frac)
    n_left = n_total
    print(f"Filling {n_total} events from {len(files)} files")

    rng = np.random.default_rng(seed)
    schema = None
    writers = {}
    buffers = {}

    for data in iter_row_groups(files, columns, weighter):
        if schema is None:
            # The first row group defines the output schema.  Every input has
            # been through _normalize_p4 and the same column selection, so the
            # schema is stable across files.
            schema = _to_plain_table(data).schema
            for split in ("train", "test"):
                writers[split] = ChunkedParquetWriter(
                    output_dir, short_name, split, schema, chunk_size, row_group_size
                )
                buffers[split] = FillBuffer(writers[split], chunk_size, rng)

        test_mask = _draw_test_mask(len(data), n_test_left, n_left, rng)
        n_test_left -= int(np.count_nonzero(test_mask))
        n_left -= len(data)

        buffers["test"].append(data[test_mask])
        buffers["train"].append(data[~test_mask])

    if schema is None:
        raise RuntimeError(f"No jets found in any of the inputs in {input_dir}")

    for split in ("train", "test"):
        buffers[split].flush()
    train_paths = writers["train"].finalize()
    test_paths = writers["test"].finalize()

    print(
        f"N={n_total}, "
        f"Ntrain={writers['train'].total_rows} in {len(train_paths)} file(s), "
        f"Ntest={writers['test'].total_rows} in {len(test_paths)} file(s) "
        f"(max {chunk_size} events per file, {row_group_size} per row group)"
    )
    return train_paths, test_paths


if __name__ == "__main__":
    sys.setrecursionlimit(50000)  # awkward concatenation can recurse deeply
    args = docopt(__doc__)

    sample_shortname = args["-s"]
    input_dir = args["-i"]
    train_frac = float(args["-f"])
    output_dir = args["-o"]
    chunk_size = int(args["--chunk-size"] or 100_000)
    row_group_size = int(args["--row-group-size"] or DEFAULT_ROW_GROUP_SIZE)
    weights_dir = args["-w"]
    side = args["--side"]
    seed = None if args["--seed"] in (None, "None") else int(args["--seed"])

    weighter = load_weighter(weights_dir, side)

    os.makedirs(output_dir, exist_ok=True)

    # Drop chunk files left over from an earlier run: a smaller dataset (or a
    # larger --chunk-size) produces fewer files, and the surplus would
    # otherwise be picked up by the globs in the downstream stages.
    for stale in glob.glob(os.path.join(output_dir, f"{sample_shortname}_*_*.parquet")):
        os.remove(stale)

    fill_splits(
        input_dir,
        output_dir,
        sample_shortname,
        train_frac,
        chunk_size=chunk_size,
        row_group_size=row_group_size,
        weighter=weighter,
        seed=seed,
    )
