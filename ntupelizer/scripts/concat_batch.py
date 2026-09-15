"""Concatenate the per-file parquets of one ntupelize batch into a single file.

Usage:
    concat_batch.py -i <input_dir> -o <output_path>

Options:
    -i <input_dir>    Directory holding the per-file .parquet outputs of one
                      ntupelize batch. All *.parquet in it are concatenated, in
                      sorted order.
    -o <output_path>  The single .parquet to write.

The files are streamed one at a time rather than concatenated in memory: for a
large sample a batch of 20 files can exceed the job's memory if held all at
once, so only one table plus the open writer is resident.

The first file fixes the output schema and every later file must match it, which
is a real constraint rather than a formality -- the per-file schema can drift if
a fill/dummy value is not in exactly the same schema as the real values it sits
beside. When that happens this reports which file diverged and how, instead of
letting pyarrow raise a bare table-vs-file schema dump.
"""

import glob
import sys

import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
from docopt import docopt


def as_array(x):
    """Return an ak.Array whether the file was written from an Array or a Record.

    ak.from_parquet gives back an ak.Record for files written with
    ak.to_parquet(ak.Record(...)); converting makes the schema consistent
    regardless of how the file was produced.
    """
    if isinstance(x, ak.Record):
        return ak.Array({k: x[k] for k in x.fields})
    return x


def to_plain_table(arr):
    """Convert to an arrow table with awkward's field metadata stripped.

    ParquetWriter compares schemas without metadata, but stripping it keeps the
    comparison -- and any error message about it -- readable.
    """
    table = ak.to_arrow_table(arr, extensionarray=False)
    clean_schema = pa.schema(
        [f.with_metadata(None) for f in table.schema], metadata=None
    )
    return table.cast(clean_schema)


def read_table(path):
    return to_plain_table(as_array(ak.from_parquet(path)))


def describe_mismatch(path, reference_path, table, schema):
    """Name the fields whose type differs, so the offending column is obvious."""
    lines = [
        f"Schema mismatch: {path}",
        f"does not match the schema taken from {reference_path}.",
    ]
    names = list(dict.fromkeys(list(schema.names) + list(table.schema.names)))
    for name in names:
        want = schema.field(name).type if name in schema.names else "<absent>"
        got = table.schema.field(name).type if name in table.schema.names else "<absent>"
        if str(want) != str(got):
            lines.append(f"  {name}:")
            lines.append(f"      expected: {want}")
            lines.append(f"      found   : {got}")
    return "\n".join(lines)


def merge(input_dir, output_path):
    files = sorted(glob.glob(f"{input_dir}/*.parquet"))
    if not files:
        sys.exit("No per-file parquets produced for this batch")

    schema = read_table(files[0]).schema
    writer = pq.ParquetWriter(
        output_path,
        schema,
        compression="zstd",
        compression_level=6,
        use_byte_stream_split=True,
    )
    try:
        for path in files:
            table = read_table(path)
            if not table.schema.equals(schema, check_metadata=False):
                sys.exit(describe_mismatch(path, files[0], table, schema))
            writer.write_table(table)
    finally:
        writer.close()
    print(f"Merged {len(files)} file(s) into {output_path}")


def main():
    args = docopt(__doc__)
    merge(args["-i"], args["-o"])


if __name__ == "__main__":
    main()
