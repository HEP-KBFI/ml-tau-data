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
