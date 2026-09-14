import glob
import os
import re

import awkward as ak
import boost_histogram as bh
import numpy as np
import vector

DUMMY_P4_VECTOR = vector.awk(
    ak.zip(
        {
            "mass": [0.0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
        }
    )
)[0]


def reinitialize_p4(p4_obj: ak.Array):
    """Reinitialized the 4-momentum for particle in order to access its properties.

    Args:
        p4_obj : ak.Array
            The particle represented by its 4-momenta

    Returns:
        p4 : ak.Array
            Particle with initialized 4-momenta.
    """
    # Initialize from all the p4 fields
    name_map = {
        "x": "px",
        "y": "py",
        "z": "pz",
        "tau": "mass",
        "t": "energy",
        "rho": "pt",
    }
    # Two source fields can map onto the same coordinate name -- e.g. a record
    # carrying both `rho` and `pt`, which is what a union of vector's internal
    # schema ({rho, phi, eta, t}) and the literal {pt, eta, phi, energy} schema
    # looks like after awkward merges them.  Building the zip straight from a
    # dict comprehension would silently let the later field win, and in such a
    # union the later field is the all-null one, so every `pt` and `energy`
    # would come out null and `ak.fill_none(..., 0.0)` downstream would turn
    # them into a column of zeros that looks like real data.  Refuse instead:
    # a collision always means the input schema is wrong upstream.
    coordinates = {}
    for field in p4_obj.fields:
        name = name_map.get(field, field)
        if name in coordinates:
            raise ValueError(
                f"Ambiguous p4 schema {p4_obj.fields}: fields "
                f"{coordinates[name]!r} and {field!r} both map to the "
                f"coordinate {name!r}. This is usually a union of two p4 "
                f"representations that should have been written as one."
            )
        coordinates[name] = field
    p4 = vector.awk(
        ak.zip({name: p4_obj[field] for name, field in coordinates.items()})
    )
    # Now make it so that the 4-vector is always saved in a similar fashion:
    p4 = vector.awk(
        ak.zip(
            {
                "pt": p4.pt,
                "eta": p4.eta,
                "phi": p4.phi,
                "energy": p4.t,
            }
        )
    )
    return p4


# Canonical zero p4 in the standardised schema, i.e. the one `reinitialize_p4`
# produces.  Note that vector stores those coordinates under its own field names
# (rho, phi, eta, t), so a literal {pt, eta, phi, energy} record is NOT the same
# schema: mixing the two in one awkward array builds a six-field union in which
# every entry is null on one side of the mix.  Build the dummy through
# `reinitialize_p4` so it always tracks the real p4s, and use it as the
# fill/dummy value wherever matched and unmatched objects share an array.
DUMMY_P4_STANDARD = reinitialize_p4(
    ak.zip(
        {
            "pt": [0.0],
            "eta": [0.0],
            "phi": [0.0],
            "energy": [0.0],
        }
    )
)[0]


def get_jet_constituent_property(property_, constituent_idx, num_ptcls_per_jet):
    reco_property_flat = property_[ak.flatten(constituent_idx, axis=-1)]
    return ak.from_iter(
        [
            ak.unflatten(reco_property_flat[i], num_ptcls_per_jet[i], axis=-1)
            for i in range(len(num_ptcls_per_jet))
        ]
    )


def to_bh(data: ak.Array, bins: np.ndarray, cumulative: bool = False) -> bh.Histogram:
    """Convert data to boost_histogram.

    Args:
        data: Data array to histogram
        bins: Bin edges
        cumulative: If True, return cumulative histogram

    Returns:
        boost_histogram Histogram object
    """
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1


def deltaphi(phi1: ak.Array, phi2: ak.Array) -> ak.Array:
    """Compute delta phi between two angles.

    Args:
        phi1: First phi angle(s)
        phi2: Second phi angle(s)

    Returns:
        Delta phi in range [-pi, pi]
    """
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


def get_all_paths(input_loc, n_files: int = None) -> list:
    """Loads all .parquet files specified by the input. The input can be a list of input_paths, a directory where the
    files are located or a wildcard path.

    Parameters:
        input_loc : str
            Location of the .parquet files.
        n_files : int
            [default: None] Maximum number of input files to be loaded. By default all will be loaded.
        columns : list
            [default: None] Names of the columns/branches to be loaded from the .parquet file. By default all columns
            will be loaded

    Returns:
        input_paths : list
            List of all the .parquet files found in the input location
    """
    if n_files == -1:
        n_files = None
    if isinstance(input_loc, list):
        input_paths = input_loc[:n_files]
    elif isinstance(input_loc, str):
        if os.path.isdir(input_loc):
            input_loc = os.path.expandvars(input_loc)
            input_paths = glob.glob(os.path.join(input_loc, "*.parquet"))[:n_files]
        elif "*" in input_loc:
            input_paths = glob.glob(input_loc)[:n_files]
        elif os.path.isfile(input_loc):
            input_paths = [input_loc]
        else:
            raise ValueError(f"Unexpected input_loc: {input_loc}")
    else:
        raise ValueError(f"Unexpected input_loc: {input_loc}")
    return input_paths


def natural_key(path: str) -> tuple:
    """Sort key that orders trailing chunk indices numerically.

    Chunk indices are zero-padded (z_train_00000.parquet), so a lexicographic
    sort is already correct for them; this key additionally stays correct if a
    file series is ever written unpadded, where plain sorting would place _10
    before _2.
    """
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", os.path.basename(path))
    )


def split_chunk_paths(input_loc: str, split: str = None) -> list:
    """Resolve a set of .parquet inputs to an ordered list of paths.

    Parameters:
        input_loc : str
            Either a directory of .parquet files, or a single .parquet file.
        split : str
            When input_loc is a directory: select only the chunks of this split
            (files named <short_name>_<split>_<index>.parquet). Pass None to
            take every .parquet in the directory, which is what the raw
            ntupelized batch directories need.

    Returns:
        paths : list
            Paths in natural (numeric) order. A single input file is returned as
            a one-element list, so callers keep working with unchunked inputs.
    """
    if os.path.isdir(input_loc):
        pattern = os.path.join(
            input_loc, f"*_{split}_*.parquet" if split else "*.parquet"
        )
        paths = sorted(glob.glob(pattern), key=natural_key)
        if not paths:
            raise FileNotFoundError(f"No .parquet files matching {pattern}")
        return paths
    if os.path.isfile(input_loc):
        return [input_loc]
    raise FileNotFoundError(f"No such file or directory: {input_loc}")


def load_parquet(input_path: str, columns: list = None) -> ak.Array:
    """Loads the contents of the .parquet file specified by the input_path

    Args:
        input_path : str
            The path to the .parquet file to be loaded.
        columns : list
            Names of the columns/branches to be loaded from the .parquet file

    Returns:
        input_data : ak.Array
            The data from the .parquet file
    """
    ret = ak.from_parquet(input_path, columns=columns)
    ret = ak.Array({k: ret[k] for k in ret.fields})
    return ret


def load_all_data(input_loc, n_files=-1, columns=None) -> ak.Array:
    if n_files == -1:
        n_files = None
    input_paths = get_all_paths(input_loc=input_loc)[:n_files]
    input_data = []
    for path in input_paths:
        input_data.append(load_parquet(path, columns=columns))
    if len(input_data) > 0:
        data = ak.concatenate(input_data)
        print("Input data loaded")
    else:
        raise ValueError(f"No files found in {input_loc}")
    return data
