import awkward as ak
import numpy as np


def compute_met(particle_data: ak.Array) -> ak.Array:
    """Missing transverse energy from all particles.

    MET = | -sum(p_T) |, the magnitude of the negative vector sum of all particle transverse momenta.

    Args:
        particle_data: [events, particles] with part_px, part_py fields

    Returns:
        [events] scalar MET values
    """
    # TODO: implement actual MET calculation
    return ak.zeros_like(ak.num(particle_data.part_px, axis=0), dtype=np.float64)


def compute_missing_pt(particle_data: ak.Array) -> ak.Array:
    """Missing pT (same as MET for now, kept separate for flexibility).

    Args:
        particle_data: [events, particles] with part_px, part_py fields

    Returns:
        [events] scalar missing pT values
    """
    # TODO: implement actual missing pt calculation
    return ak.zeros_like(ak.num(particle_data.part_px, axis=0), dtype=np.float64)


def compute_num_jets(jet_data: ak.Array) -> ak.Array:
    """Number of jets per event.

    Args:
        jet_data: [events, jets] with jet_pt (or any jet-level field)

    Returns:
        [events] integer jet multiplicity per event
    """
    return ak.num(jet_data.jet_pt, axis=-1)


def compute_ht(jet_data: ak.Array) -> ak.Array:
    """HT = scalar sum of jet pT.

    Args:
        jet_data: [events, jets] with jet_pt field

    Returns:
        [events] HT values
    """
    return ak.sum(jet_data.jet_pt, axis=-1)


def compute_sum_et(particle_data: ak.Array) -> ak.Array:
    """Sum of transverse energy of all particles in the event.

    Args:
        particle_data: [events, particles] with part_pt (or part_energy * sin(theta)) fields

    Returns:
        [events] scalar sum of E_T
    """
    # TODO: verify this is the correct E_T definition for ALEPH
    return ak.sum(particle_data.part_pt, axis=-1)


def compute_thrust(particle_data: ak.Array) -> ak.Array:
    """Event thrust T = max over n of sum_i |p_i . n| / sum_i |p_i|.

    Args:
        particle_data: [events, particles] with part_px, part_py, part_pz fields

    Returns:
        [events] thrust values (between 0.5 and 1.0)
    """
    # TODO: implement actual thrust calculation
    return ak.zeros_like(ak.num(particle_data.part_px, axis=0), dtype=np.float64)


def compute_aplanarity(particle_data: ak.Array) -> ak.Array:
    """Event aplanarity from the momentum tensor eigenvalues.

    Aplanarity = 3/2 * lambda_3 where lambda_3 is the smallest eigenvalue
    of the normalized momentum tensor.

    Args:
        particle_data: [events, particles] with part_px, part_py, part_pz fields

    Returns:
        [events] aplanarity values
    """
    # TODO: implement actual aplanarity calculation
    return ak.zeros_like(ak.num(particle_data.part_px, axis=0), dtype=np.float64)


def compute_sphericity(particle_data: ak.Array) -> ak.Array:
    """Event sphericity from the momentum tensor eigenvalues.

    Sphericity = 3/2 * (lambda_2 + lambda_3) where lambda_i are the
    eigenvalues of the normalized momentum tensor.

    Args:
        particle_data: [events, particles] with part_px, part_py, part_pz fields

    Returns:
        [events] sphericity values (between 0 and 1)
    """
    # TODO: implement actual sphericity calculation
    return ak.zeros_like(ak.num(particle_data.part_px, axis=0), dtype=np.float64)


def compute_particle_counts(particle_data: ak.Array) -> ak.Array:
    """Counts of different particle types per event.

    Args:
        particle_data: [events, particles] with part_isElectron, part_isMuon,
                       part_isPhoton, part_isChargedHadron, part_isNeutralHadron fields

    Returns:
        ak.Array with fields num_electrons, num_muons, num_photons,
                 num_charged_hadrons, num_neutral_hadrons, each of shape [events]
    """
    return ak.zip(
        {
            "num_electrons": ak.sum(particle_data.part_isElectron, axis=-1),
            "num_muons": ak.sum(particle_data.part_isMuon, axis=-1),
            "num_photons": ak.sum(particle_data.part_isPhoton, axis=-1),
            "num_charged_hadrons": ak.sum(particle_data.part_isChargedHadron, axis=-1),
            "num_neutral_hadrons": ak.sum(particle_data.part_isNeutralHadron, axis=-1),
        }
    )


def get_event_variables(particle_data: ak.Array, jet_data: ak.Array) -> dict:
    """Compute all event-level variables and return as a flat dict of [events] arrays.

    Args:
        particle_data: [events, particles] full particle record array
        jet_data: [events, jets] jet-level data (jet_pt, etc.)

    Returns:
        dict of field_name -> [events] arrays
    """
    particle_counts = compute_particle_counts(particle_data)
    return {
        "event_met": compute_met(particle_data),
        "event_missing_pt": compute_missing_pt(particle_data),
        "event_num_jets": compute_num_jets(jet_data),
        "event_ht": compute_ht(jet_data),
        "event_sum_et": compute_sum_et(particle_data),
        "event_thrust": compute_thrust(particle_data),
        "event_aplanarity": compute_aplanarity(particle_data),
        "event_sphericity": compute_sphericity(particle_data),
        "event_num_electrons": particle_counts.num_electrons,
        "event_num_muons": particle_counts.num_muons,
        "event_num_photons": particle_counts.num_photons,
        "event_num_charged_hadrons": particle_counts.num_charged_hadrons,
        "event_num_neutral_hadrons": particle_counts.num_neutral_hadrons,
    }
