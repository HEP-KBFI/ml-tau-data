import hydra
import uproot
import vector
import numpy as np
import awkward as ak

from omegaconf import DictConfig
from typing import Optional, List
from ntupelizer.tools import features as f


def load_file_contents(
    path: str, tree_name: str = "events", branches: Optional[List[str]] = None
) -> ak.Array:
    with uproot.open(path) as in_file:
        tree = in_file[tree_name]
        arrays = tree.arrays(branches)
    return arrays


def fill_values(array: ak.Array, fill_value: float = -999.9):
    return ak.fill_none(ak.pad_none(array, 1, axis=-1), fill_value)


def calculate_impact_parameters(
    events: ak.Array, track_states_collection: str = "_Tracks_trackStates"
):

    ##
    ## Here, with fill values, one should have the same # of entries as there are particles.
    ## Therefore a single 0.0 is not enough. Check the shapes.
    ##
    d0 = fill_values(events[f"{track_states_collection}.D0"])
    z0 = fill_values(events[f"{track_states_collection}.Z0"])
    phi0 = fill_values(events[f"{track_states_collection}.phi"])
    tanL = fill_values(events[f"{track_states_collection}.tanLambda"])
    # omega = fill_values(events[f"{track_states_collection}.omega"])
    xr = fill_values(events[f"{track_states_collection}.referencePoint.x"])
    yr = fill_values(events[f"{track_states_collection}.referencePoint.y"])
    zr = fill_values(events[f"{track_states_collection}.referencePoint.z"])

    # Extract covariance matrix elements
    cov_matrix = events[f"{track_states_collection}.covMatrix.values[21]"]
    d0_error = fill_values(cov_matrix[:, :, 0])
    phi0_error = fill_values(cov_matrix[:, :, 2])
    z0_error = fill_values(cov_matrix[:, :, 9])
    tanL_error = fill_values(cov_matrix[:, :, 14])

    # Vertex position (assuming you have this defined)
    vertex_x = ak.flatten(fill_values(events["Vertices.position.x"]), axis=-1)
    vertex_y = ak.flatten(fill_values(events["Vertices.position.y"]), axis=-1)
    vertex_z = ak.flatten(fill_values(events["Vertices.position.z"]), axis=-1)

    # Bare in mind that Vertices are not found for some events ... need to find out why
    ##################

    # Calculate track origins (vectorized) - from lifetime.py
    x0 = xr + np.cos(np.pi / 2 - phi0) * d0
    y0 = yr - np.sin(np.pi / 2 - phi0) * d0
    z0_prime = z0 + zr

    # Calculate PCA arc length (vectorized) - simplified 3D version
    # Following the pattern from calc_pca_arc_length_vectorized
    ax, ay, az = x0, y0, z0_prime
    bx = np.cos(phi0) + x0
    by = np.sin(phi0) + y0
    bz = tanL + z0_prime

    num = -(
        (ax - vertex_x) * (bx - ax)
        + (ay - vertex_y) * (by - ay)
        + (az - vertex_z) * (bz - az)
    )
    den = (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2
    s = num / den

    # Calculate PCA position (vectorized)
    pca_x = s * np.cos(phi0) + x0
    pca_y = s * np.sin(phi0) + y0
    pca_z = s * tanL + z0_prime

    # Calculate impact parameters
    dxy = np.sqrt((vertex_x - pca_x) ** 2 + (vertex_y - pca_y) ** 2)
    dz = np.abs(pca_z - vertex_z)

    # Calculate errors for PCA position (simplified error propagation)
    # Following pattern from calc_pca_position_error_vectorized
    s_error = np.sqrt(d0_error**2 + phi0_error**2)  # Simplified estimate

    x0_error = np.sqrt(
        (np.cos(np.pi / 2 - phi0) * d0_error) ** 2
        + (np.sin(np.pi / 2 - phi0) * d0 * phi0_error) ** 2
    )
    y0_error = np.sqrt(
        (np.sin(np.pi / 2 - phi0) * d0_error) ** 2
        + (np.cos(np.pi / 2 - phi0) * d0 * phi0_error) ** 2
    )
    z0_prime_error = z0_error

    pca_x_error = np.sqrt(
        (np.cos(phi0) * s_error) ** 2
        + x0_error**2
        + (s * np.sin(phi0) * phi0_error) ** 2
    )
    pca_y_error = np.sqrt(
        (np.sin(phi0) * s_error) ** 2
        + y0_error**2
        + (s * np.cos(phi0) * phi0_error) ** 2
    )
    pca_z_error = np.sqrt(
        (tanL * s_error) ** 2 + (s * tanL_error) ** 2 + z0_prime_error**2
    )

    # Calculate impact parameter errors
    dxy_error = np.sqrt(
        ((vertex_x - pca_x) / dxy * pca_x_error) ** 2
        + ((vertex_y - pca_y) / dxy * pca_y_error) ** 2
    )
    dz_error = pca_z_error
    return ak.zip(
        {
            "dzval": dz,
            "dzerr": dz_error,
            "d0val": d0,
            "d0err": d0_error,
            "dxy": dxy,
            "dxyerr": dxy_error,
        }
    )


def assign_pid_info(arrays: ak.Array) -> ak.Array:
    """Extract particle ID info vectorized across all events"""
    # 0:Charged hadrons, 1:Electron, 2:Muon, 3:Track from V0, 4:photons, 5:Neutral hadrons, 6:Hcalobject, 7:Lcalobject
    pid_type = arrays["ParticleID.type"]
    return ak.zip(
        {
            "is_charged_hadron": ak.values_astype(pid_type == 0, "int32"),
            "is_electron": ak.values_astype(pid_type == 1, "int32"),
            "is_muon": ak.values_astype(pid_type == 2, "int32"),
            "is_photon": ak.values_astype(pid_type == 4, "int32"),
            "is_neutral_hadron": ak.values_astype(pid_type == 5, "int32"),
        }
    )


def pad_missing_values(
    pad_target: ak.Array, pad_from: ak.Array, pad_value: float = -999.9
):
    n_large = ak.num(pad_from)
    n_small = ak.num(pad_target)
    pad_counts = n_large - n_small
    padding = ak.unflatten(
        ak.full_like(np.zeros(ak.sum(pad_counts)), pad_value), pad_counts
    )
    return ak.concatenate([pad_target, padding], axis=1)


def get_cand_info(events: ak.Array) -> ak.Array:
    """Extract and combine info from multiple collections vectorized"""
    # Extract ReconstructedParticles info (all events at once)
    cand_p4 = vector.awk(
        ak.zip(
            {
                "energy": events["RecoParticles.energy"],
                "px": events["RecoParticles.momentum.x"],
                "py": events["RecoParticles.momentum.y"],
                "pz": events["RecoParticles.momentum.z"],
            }
        )
    )

    # Get PID info
    pid_info = assign_pid_info(events)
    impact_parameters = calculate_impact_parameters(events=events)
    impact_parameters_choice = find_linked_indices(events=events)
    impact_parameters = impact_parameters[impact_parameters_choice]
    # TODO: Check that impact parameters are assigned correctly.
    impact_parameters = ak.zip(
        {
            field: pad_missing_values(
                pad_target=impact_parameters[field],
                pad_from=pid_info["is_charged_hadron"],
                pad_value=-999.9,
            )
            for field in impact_parameters.fields
        }
    )

    # Combine everything into one structure (all vectorized)
    particle_data = ak.zip(
        {
            "part_p4": cand_p4,
            "part_px": cand_p4.px,
            "part_py": cand_p4.py,
            "part_pz": cand_p4.pz,
            "part_energy": cand_p4.energy,
            "part_pt": cand_p4.pt,
            "part_eta": cand_p4.eta,
            "part_phi": cand_p4.phi,
            "part_mass": cand_p4.mass,
            "part_d0val": impact_parameters["d0val"],
            "part_d0err": impact_parameters["d0err"],
            "part_dzval": impact_parameters["dzval"],
            "part_dzerr": impact_parameters["dzerr"],
            "part_charge": events["RecoParticles.charge"],
            "part_isChargedHadron": pid_info["is_charged_hadron"],
            "part_isNeutralHadron": pid_info["is_neutral_hadron"],
            "part_isPhoton": pid_info["is_photon"],
            "part_isElectron": pid_info["is_electron"],
            "part_isMuon": pid_info["is_muon"],
        }
    )
    return particle_data


def find_linked_indices(
    events: ak.Array,
    begin_branch: str = "RecoParticles.tracks_begin",
    end_branch: str = "RecoParticles.tracks_end",
):
    begins = events[begin_branch]
    ends = events[end_branch]
    counts = ends - begins
    flat_counts = ak.flatten(counts)
    total = ak.sum(flat_counts)
    base = np.zeros(total, dtype=int)
    local = ak.unflatten(base, flat_counts)
    local = ak.local_index(local)
    local = ak.unflatten(local, ak.num(counts))
    indices = begins + local
    return ak.flatten(indices, axis=-1)


def get_jet_basic_properties(events: ak.Array):
    jet_p4 = vector.awk(
        ak.zip(
            {
                "energy": events["Jets.energy"],
                "px": events["Jets.momentum.x"],
                "py": events["Jets.momentum.y"],
                "pz": events["Jets.momentum.z"],
            }
        )
    )
    constituents_begins = events["Jets.particles_begin"]
    constituents_ends = events["Jets.particles_end"]
    constituent_counts = constituents_ends - constituents_begins

    jet_data = ak.zip(
        {
            "jet_pt": jet_p4.pt,
            "jet_eta": jet_p4.eta,
            "jet_phi": jet_p4.phi,
            "jet_energy": jet_p4.energy,
            "jet_nparticles": constituent_counts,
            "jet_sdmass": ak.ones_like(jet_p4.pt)
            * -999.9,  # Softdrop mass. SDmass. For the moment added a placeholder
            "jet_tau1": ak.ones_like(jet_p4.pt)
            * -999.9,  # N-subjettiness variable. For the moment added a placeholder
            "jet_tau2": ak.ones_like(jet_p4.pt)
            * -999.9,  # N-subjettiness variable. For the moment added a placeholder
            "jet_tau3": ak.ones_like(jet_p4.pt)
            * -999.9,  # N-subjettiness variable. For the moment added a placeholder
            "jet_tau4": ak.ones_like(jet_p4.pt)
            * -999.9,  # N-subjettiness variable. For the moment added a placeholder
        }
    )  # TODO For the last 5 variables need to define a calculator function.
    return jet_data


def get_all_properties(events: ak.Array, particle_data: ak.Array, jet_data: ak.Array):
    jc_indices = find_linked_indices(
        events=events,
        begin_branch="Jets.particles_begin",
        end_branch="Jets.particles_end",
    )
    # Assign the particles with their properties to the jets they belong to
    jet_constituent_properties = particle_data[jc_indices]
    flat_particles = ak.flatten(jet_constituent_properties)
    flat_counts = ak.flatten(jet_data.jet_nparticles, axis=None)
    true_counts = ak.num(jet_data.jet_nparticles)
    jet_wise_particles = ak.unflatten(flat_particles, flat_counts)
    jet_assigned_particles = ak.unflatten(jet_wise_particles, true_counts)

    jet_constituent_p4_sums = ak.sum(jet_assigned_particles.part_p4, axis=-1)

    all_properties = ak.Array(
        {
            "part_ptrel": jet_assigned_particles.part_pt / jet_data.jet_pt,
            "part_erel": jet_assigned_particles.part_energy / jet_data.jet_energy,
            "part_etarel": jet_assigned_particles.part_eta
            - jet_data.jet_eta,  # Honestly, what is the actual difference?
            "part_phirel": jet_assigned_particles.part_phi - jet_data.jet_phi,
            "part_deltaR": f.deltaR_etaPhi(
                jet_assigned_particles.part_eta,
                jet_assigned_particles.part_phi,
                jet_data.jet_eta,
                jet_data.jet_phi,
            ),
            "part_deta": f.deltaEta(jet_assigned_particles.part_eta, jet_data.jet_eta),
            "part_dphi": f.deltaPhi(jet_assigned_particles.part_phi, jet_data.jet_phi),
            "jet_mass_from_p4s": jet_constituent_p4_sums.mass,
            "jet_pt_from_p4s": jet_constituent_p4_sums.pt,
            "jet_eta_from_p4s": jet_constituent_p4_sums.eta,
            "jet_phi_from_p4s": jet_constituent_p4_sums.phi,
            **{field: jet_data[field] for field in jet_data.fields},
            **{
                field: jet_assigned_particles[field]
                for field in jet_assigned_particles.fields
                if field != "part_p4"
            },
        }
    )
    padded = ak.pad_none(all_properties, 1, axis=-1)
    filled = ak.fill_none(padded, -999.9)
    jet_based_dataset = ak.Array(
        {field: ak.flatten(filled[field], axis=1) for field in filled.fields}
    )
    return jet_based_dataset


def construct_jet_based_dataset(events: ak.Array):
    particle_data = get_cand_info(events=events)
    jet_data = get_jet_basic_properties(events=events)
    all_properties = get_all_properties(
        events=events, particle_data=particle_data, jet_data=jet_data
    )
    return all_properties


def ntupelize_file(input_path: str, output_path: str):
    events = load_file_contents(path=input_path, tree_name="events")
    events = events[ak.num(events["Jets.energy"]) > 0]
    dataset = construct_jet_based_dataset(events)
    ak.to_parquet(dataset, output_path, row_group_size=1024)
