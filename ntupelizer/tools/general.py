import os
import glob
import vector
import numpy as np
import awkward as ak
import boost_histogram as bh


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
    p4 = vector.awk(
        ak.zip({name_map.get(field, field): p4_obj[field] for field in p4_obj.fields})
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
