import numpy as np


def get_decaymode(pdg_ids):
    """Tau decaymodes are the following:
    decay_mode_mapping = {
        0: 'OneProng0PiZero',
        1: 'OneProng1PiZero',
        2: 'OneProng2PiZero',
        3: 'OneProng3PiZero',
        4: 'OneProngNPiZero',
        5: 'TwoProng0PiZero',
        6: 'TwoProng1PiZero',
        7: 'TwoProng2PiZero',
        8: 'TwoProng3PiZero',
        9: 'TwoProngNPiZero',
        10: 'ThreeProng0PiZero',
        11: 'ThreeProng1PiZero',
        12: 'ThreeProng2PiZero',
        13: 'ThreeProng3PiZero',
        14: 'ThreeProngNPiZero',
        15: 'RareDecayMode'
        16: 'LeptonicDecay'
    }
    0: [0, 5, 10]
    1: [1, 6, 11]
    2: [2, 3, 4, 7, 8, 9, 12, 13, 14, 15]
    """
    pdg_ids = np.abs(np.array(pdg_ids))
    unique, counts = np.unique(pdg_ids, return_counts=True)
    common_particles = [16, 130, 211, 13, 14, 12, 11]
    n_uncommon = len(set(unique) - set(common_particles))
    p_counts = {i: 0 for i in common_particles}
    p_counts.update(dict(zip(unique, counts)))
    if n_uncommon > 0:
        return 15
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 0:
        return 0
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 1:
        return 1
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 2:
        return 2
    elif np.sum(p_counts[211]) == 1 and p_counts[130] == 3:
        return 3
    elif np.sum(p_counts[211]) == 1 and p_counts[130] > 3:
        return 4
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 0:
        return 5
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 1:
        return 6
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 2:
        return 7
    elif np.sum(p_counts[211]) == 2 and p_counts[130] == 3:
        return 8
    elif np.sum(p_counts[211]) == 2 and p_counts[130] > 3:
        return 9
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 0:
        return 10
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 1:
        return 11
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 2:
        return 12
    elif np.sum(p_counts[211]) == 3 and p_counts[130] == 3:
        return 13
    elif np.sum(p_counts[211]) == 3 and p_counts[130] > 3:
        return 14
    elif np.sum(p_counts[11] + p_counts[13]) > 0:
        return 16
    else:
        return 15


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
