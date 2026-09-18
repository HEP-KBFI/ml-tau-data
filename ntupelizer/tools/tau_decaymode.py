"""Tau decay mode classification, shared by ml-tau-data and ml-tau-model.

This module is the single definition of "which decay mode is this tau".  Both
the ntupelizer (writing `gen_jet_tau_decaymode`) and the model evaluation
(deriving the same quantity from a predicted or true daughter set) go through
`classify_decay_mode` here, so the two cannot drift apart.

The classification works off particle *properties* rather than a hardcoded list
of PDG ids.  A species is a prong if it is a charged hadron and a "pi0-like"
neutral if it is a neutral hadron, whatever its PDG id happens to be.  Earlier
versions enumerated the ids, and every time the generator produced something not
on the list -- K0_S, omega -- it was silently counted as nothing at all.
"""

from functools import lru_cache

import numpy as np
from particle import pdgid

# Ignored entirely when classifying: they are not decay products in the
# prong/pi0 sense.  Photons are ignored on purpose, so a radiative decay lands
# in the same class as the non-radiative one -- PDG treats tau radiative modes
# as indented sub-modes of their parent (e.g. Gamma6 e nu nu gamma under
# Gamma5 e nu nu), defined only relative to a photon energy cutoff, not as
# separate channels.
IGNORED_PDGS = frozenset({12, 14, 16, 22})  # neutrinos and photons

LEPTON_PDGS = frozenset({11, 13})

# "Normal" convention, as stored in gen_jet_tau_decaymode and as used by
# DM_NAME_MAPPING below.
RARE_DECAY_MODE = 15
# "Extended" convention.  15 is reachable on the bare 5*(n_charged-1)+n_neutral
# grid (four prongs, no neutrals), so a rare decay and a four-prong
# reconstruction are indistinguishable under the normal convention.  30 is off
# the grid, which keeps that failure mode visible in a confusion matrix.
RARE_DECAY_MODE_EXT = 30

LEPTONIC_DECAY_MODE = 16

MAX_NEUTRALS = 4  # n_neutral is clamped here: 4, 9, 14 are the "N pi0" buckets
MAX_PRONGS = 3


@lru_cache(maxsize=None)
def _classify_pdg(pdg):
    """-> 'charged' | 'neutral' | 'lepton' | 'ignored' | 'other' for one |PDG|."""
    pdg = int(abs(pdg))
    if pdg in IGNORED_PDGS:
        return "ignored"
    if pdg in LEPTON_PDGS:
        return "lepton"
    if pdgid.is_hadron(pdg):
        return "charged" if abs(pdgid.charge(pdg)) > 0 else "neutral"
    return "other"


def count_daughters(pdg_ids):
    """Count one tau's visible daughters by category.

    Returns (n_charged, n_neutral, n_lepton, n_other).  Neutrinos and photons
    are not counted at all.
    """
    counts = {"charged": 0, "neutral": 0, "lepton": 0, "other": 0, "ignored": 0}
    for pdg in pdg_ids:
        counts[_classify_pdg(pdg)] += 1
    return counts["charged"], counts["neutral"], counts["lepton"], counts["other"]


def decay_mode_from_counts(
    n_charged, n_neutral, n_lepton=0, n_other=0, rare=RARE_DECAY_MODE
):
    """Map daughter multiplicities onto a decay mode id.

    The grid is 5*(n_charged - 1) + min(n_neutral, MAX_NEUTRALS), i.e. 0-4 for
    one prong, 5-9 for two, 10-14 for three.  Anything off it -- more than three
    prongs, or a daughter that is neither hadron nor lepton -- is `rare`.  A tau
    with no charged hadron but a lepton is LEPTONIC_DECAY_MODE.
    """
    if n_other > 0:
        return rare
    if n_charged == 0:
        return LEPTONIC_DECAY_MODE if n_lepton > 0 else rare
    if n_charged > MAX_PRONGS:
        return rare
    return 5 * (n_charged - 1) + min(n_neutral, MAX_NEUTRALS)


def classify_decay_mode(pdg_ids, rare=RARE_DECAY_MODE):
    """Decay mode of one tau from the PDG ids of its visible daughters.

    Pass the raw generator PDG ids; they are categorised by property here, so
    they must NOT be pre-mapped onto representative ids.
    """
    n_charged, n_neutral, n_lepton, n_other = count_daughters(pdg_ids)
    return decay_mode_from_counts(n_charged, n_neutral, n_lepton, n_other, rare=rare)


def classify_decay_modes(pdg_ids_per_tau, rare=RARE_DECAY_MODE):
    """`classify_decay_mode` over many taus; takes a list of lists or an awkward array.

    Converts an awkward array to Python lists up front: iterating one row at a
    time is far slower than one bulk conversion, and the per-species decision is
    cached, so the cost is dominated by the conversion rather than the physics.
    """
    rows = pdg_ids_per_tau
    if hasattr(rows, "to_list"):
        rows = rows.to_list()
    elif hasattr(rows, "layout"):  # awkward array without the helper
        import awkward as ak

        rows = ak.to_list(rows)
    return np.array(
        [classify_decay_mode(row, rare=rare) for row in rows], dtype=np.int64
    )


def get_decaymode(pdg_ids):
    """Back-compatible alias for `classify_decay_mode`.

    Kept because callers pass ids already mapped onto representatives (charged
    hadrons as 211, neutral hadrons as 130).  That still classifies correctly,
    since 211 and 130 are themselves a charged and a neutral hadron, but new
    code should call `classify_decay_mode` with the raw PDG ids instead.

    decay_mode_mapping is DM_NAME_MAPPING at the bottom of this module.
    """
    return classify_decay_mode(pdg_ids)


def get_reduced_decaymodes(decaymodes: np.array):
    """Maps the full set of decay modes into a smaller subset, setting the rarer decaymodes under "Other" (# 15)"""
    target_mapping = {
        -1: 15,  # As we are running DM classification only on signal sample, then HPS_dm of -1 = 15 (Rare)
        0: 0,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 10,
        6: 11,
        7: 11,
        8: 11,
        9: 11,
        10: 10,
        11: 11,
        12: 11,
        13: 11,
        14: 11,
        15: 15,
        16: 16,
    }
    return np.vectorize(target_mapping.get)(decaymodes)


# Initial mapping
DM_NAME_MAPPING = {
    0: "OneProng0PiZero",
    1: "OneProng1PiZero",
    2: "OneProng2PiZero",
    3: "OneProng3PiZero",
    4: "OneProngNPiZero",
    5: "TwoProng0PiZero",
    6: "TwoProng1PiZero",
    7: "TwoProng2PiZero",
    8: "TwoProng3PiZero",
    9: "TwoProngNPiZero",
    10: "ThreeProng0PiZero",
    11: "ThreeProng1PiZero",
    12: "ThreeProng2PiZero",
    13: "ThreeProng3PiZero",
    14: "ThreeProngNPiZero",
    15: "RareDecayMode",
    16: "LeptonicDecay",
}
