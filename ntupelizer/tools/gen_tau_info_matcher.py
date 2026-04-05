import vector
import numpy as np
import awkward as ak
from particle import pdgid

from ntupelizer.tools import general as g
from ntupelizer.tools import matching as m
from ntupelizer.tools import tau_decaymode as dm


class GenTauInfoMatcher:
    def __init__(
        self,
        arrays: ak.Array,
        gen_jets: ak.Array,
        idx_map_branch: str = "idx_mc",
        debug: bool = False,
    ):
        """Associates the gen_tau info with the gen_jets

        Parameters:
            arrays: ak.Array
                All the data used for the preprocessing for all events
            gen_jets: ak.Array
                All the clustered gen jets
            idx_map_branch: str
                Branch name for the idx_map. The newer PodioROOT version uses "_RecoMCTruthLink_to.index"
                (renamed as idx_mc in the its ntupelizer class) and the older LCIO version uses
                "MCParticles#1.index"
        """
        self.arrays = arrays
        self.debug = debug
        self.gen_jets = gen_jets
        self.idx_map_branch = idx_map_branch
        self.properties = [
            "tau_DV_x",
            "tau_DV_y",
            "tau_DV_z",
            "tau_full_p4",
            "tau_decaymode",
            "tau_p4",  # This is the visible p4
            "tau_vis_energy",
            "tau_charge",
            "tau_daughter_PDG",
        ]
        self.fill_values = {
            "tau_vis_energy": 0,
            "tau_decaymode": -1,
            "tau_charge": -999,
            "tau_full_p4": g.DUMMY_P4_VECTOR,
            "tau_p4": g.DUMMY_P4_VECTOR,  # This is the visible p4
            "tau_DV_x": -1,
            "tau_DV_y": -1,
            "tau_DV_z": -1,
        }

    def map_pdgid_to_candid(self, pdg_id):
        if pdgid.is_hadron(pdg_id):
            if abs(pdgid.charge(pdg_id)) > 0:
                return 211  # charged hadron
            else:
                return 130  # neutral hadron
        else:
            return abs(pdg_id)

    def get_all_tau_best_combinations(self, vis_tau_p4s, gen_jets):
        vis_tau_p4s = g.reinitialize_p4(vis_tau_p4s)
        gen_jets_p4 = g.reinitialize_p4(gen_jets)
        tau_indices, gen_indices = m.dr_matching(vis_tau_p4s, gen_jets_p4, 0.4)
        pairs = []
        for tau_idx, gen_idx in zip(tau_indices, gen_indices):
            pair = []
            for i, tau_idx_i in enumerate(tau_idx):
                pair.append([tau_idx_i, gen_idx[i]])
            pairs.append(pair)
        return ak.Array(pairs)

    def get_matched_gen_tau_property(
        self, gen_jets, best_combos, property_, dummy_value=-1
    ):
        gen_jet_full_info_array = []
        for event_id, event_gen_jets in enumerate(gen_jets):
            mapping = {i[1]: i[0] for i in best_combos[event_id]}
            gen_jet_info_array = []
            for i, _ in enumerate(event_gen_jets):
                if len(best_combos[event_id]) > 0:
                    if i in best_combos[event_id][:, 1]:
                        value = property_[event_id][mapping[i]]
                        gen_jet_info_array.append(value)
                    else:
                        gen_jet_info_array.append(dummy_value)
                else:
                    gen_jet_info_array.append(dummy_value)
            gen_jet_full_info_array.append(gen_jet_info_array)
        return ak.Array(gen_jet_full_info_array)

    def retrieve_tau_info_from_daughters(
        self, tau_daughters, n_taus, event, event_particle_p4s
    ):
        tau_decay_modes = []
        tau_vis_p4s = []
        daughter_pdgs = []
        for tau_idx in range(n_taus):
            daughter_pdgs = [
                event["MCParticles.PDG"][d_idx] for d_idx in tau_daughters[tau_idx]
            ]
            pdgs = [self.map_pdgid_to_candid(pdg_id) for pdg_id in daughter_pdgs]
            tau_vis_p4 = g.DUMMY_P4_VECTOR
            for tc in tau_daughters[tau_idx]:
                daughter_p4 = event_particle_p4s[tc]
                if abs(event["MCParticles.PDG"][tc]) not in [12, 14, 16]:
                    tau_vis_p4 = tau_vis_p4 + daughter_p4
            tau_vis_p4s.append(tau_vis_p4)
            tau_decay_modes.append(dm.get_decaymode(pdgs))
            daughter_pdgs.append(pdgs)
        tau_vis_p4s = g.reinitialize_p4(ak.Array(tau_vis_p4s))
        tau_info = {
            "tau_decaymode": ak.Array(tau_decay_modes),
            "tau_p4": tau_vis_p4s,
            "tau_daughter_PDG": ak.Array(daughter_pdgs),
            "tau_vis_energy": tau_vis_p4s.energy,
        }
        return tau_info

    def retrieve_tau_general_info(
        self, tau_indices: np.array, event: ak.Array, event_particle_p4s: ak.Array
    ):
        tau_general_info = {}
        tau_general_info["tau_DV_x"] = [
            event["MCParticles.endpoint.x"][tau_idx] for tau_idx in tau_indices
        ]
        tau_general_info["tau_DV_y"] = [
            event["MCParticles.endpoint.y"][tau_idx] for tau_idx in tau_indices
        ]
        tau_general_info["tau_DV_z"] = [
            event["MCParticles.endpoint.z"][tau_idx] for tau_idx in tau_indices
        ]
        tau_general_info["tau_full_p4"] = [
            event_particle_p4s[tau_idx] for tau_idx in tau_indices
        ]
        tau_general_info["tau_charge"] = [
            pdgid.charge(event["MCParticles.PDG"][tau_idx]) for tau_idx in tau_indices
        ]
        return tau_general_info

    def fill_tau_info(self):
        tau_info = {property: [] for property in self.properties}
        for event_idx, event in enumerate(self.arrays):
            d_idx = event["_MCParticles_daughters.index"]
            d_begin = event["MCParticles.daughters_begin"]
            d_end = event["MCParticles.daughters_end"]
            tau_mask = (np.abs(event["MCParticles.PDG"]) == 15) * (
                event["MCParticles.generatorStatus"] == 2
            )
            tau_indices = np.where(tau_mask)[0]
            tau_daughters = []
            if self.debug:
                print("---------")
                print("Event no.: ", event_idx)
            for tau_idx in tau_indices:
                daughter_indices = d_idx[d_begin[tau_idx] : d_end[tau_idx]]
                if self.debug:
                    print("Tau_idx: ", tau_idx)
                    print(event["MCParticles.PDG"][daughter_indices])
                tau_daughters.append(daughter_indices)
            event_particle_p4s = vector.awk(
                ak.zip(
                    {
                        "mass": event["MCParticles.mass"],
                        "x": event["MCParticles.momentum.x"],
                        "y": event["MCParticles.momentum.y"],
                        "z": event["MCParticles.momentum.z"],
                    }
                )
            )
            tau_general_info = self.retrieve_tau_general_info(
                tau_indices, event, event_particle_p4s
            )
            tau_daughter_info = self.retrieve_tau_info_from_daughters(
                tau_daughters, len(tau_indices), event, event_particle_p4s
            )
            for key, value in tau_general_info.items():
                tau_info[key].append(value)
            for key, value in tau_daughter_info.items():
                tau_info[key].append(value)
        merged_tau_info = {key: ak.Array(value) for key, value in tau_info.items()}
        best_combos = self.get_all_tau_best_combinations(
            merged_tau_info["tau_p4"], self.gen_jets
        )
        gen_jet_tau_info = {}
        for key, dummy_value in self.fill_values.items():
            gen_jet_tau_info[f"gen_jet_{key}"] = self.get_matched_gen_tau_property(
                self.gen_jets,
                best_combos,
                merged_tau_info[key],
                dummy_value=dummy_value,
            )
        return gen_jet_tau_info
