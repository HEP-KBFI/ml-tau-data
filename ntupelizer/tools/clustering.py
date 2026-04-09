import vector
import fastjet
import numpy as np
import awkward as ak
from ntupelizer.tools import general as g


class JetClusterer:
    def __init__(
        self,
        particles: ak.Array,
        particles_p4: ak.Array,
        min_pt: float = 0.0,
        deltar: float = 0.4,
    ):
        self.min_pt = min_pt
        self.deltar = deltar
        self.particles, self.particles_p4 = particles, particles_p4
        self.all_jets, self.all_constituent_indices = self._cluster_jets()
        self.jets, self.constituent_indices = self._filter_jets()

    def _filter_jets(self):
        raise NotImplementedError("Please implement this function for your sub-class")

    def _cluster_jets(self):
        jetdef = fastjet.JetDefinition2Param(
            fastjet.ee_genkt_algorithm, self.deltar, -1
        )
        cluster = fastjet.ClusterSequence(self.particles_p4, jetdef)
        jets = vector.awk(cluster.inclusive_jets(min_pt=self.min_pt))
        jets = vector.awk(
            ak.zip(
                {"energy": jets["t"], "x": jets["x"], "y": jets["y"], "z": jets["z"]}
            )
        )
        constituent_index = ak.Array(cluster.constituent_index(min_pt=self.min_pt))
        njets = np.sum(ak.num(jets))
        print(f"clustered {njets} jets")
        return jets, constituent_index

    @property
    def results(self):
        return self.jets, self.constituent_indices


class RecoJetClusterer(JetClusterer):
    def __init__(
        self,
        particles: ak.Array,
        particles_p4: ak.Array,
        min_pt: float = 0.0,
        deltar: float = 0.4,
    ):
        super().__init__(
            particles=particles, particles_p4=particles_p4, min_pt=min_pt, deltar=deltar
        )

    def _filter_jets(self):
        """Actually one should also consider removing jets that have lepton near it or a Z boson."""
        return self.all_jets, self.all_constituent_indices


class GenJetClusterer(JetClusterer):
    def __init__(
        self,
        particles: ak.Array,
        particles_p4: ak.Array,
        min_pt: float = 0.0,
        deltar: float = 0.4,
    ):
        super().__init__(
            particles=particles, particles_p4=particles_p4, min_pt=min_pt, deltar=deltar
        )

    def _filter_jets(self):
        """Filter out all gen jets that have a lepton as one of their consituents (so in dR < 0.4)
        Currently see that also some jets with 6 hadrons and an electron are filtered out
        Roughly 90% of gen jets will be left after filtering
        """
        gen_num_ptcls_per_jet = ak.num(self.all_constituent_indices, axis=-1)
        gen_jet_pdgs = g.get_jet_constituent_property(
            self.particles.PDG, self.all_constituent_indices, gen_num_ptcls_per_jet
        )
        mask = []
        for gj_pdg in gen_jet_pdgs:
            sub_mask = []
            for gjp in gj_pdg:
                if (15 in np.abs(gjp)) or (13 in np.abs(gjp)):
                    sub_mask.append(False)
                else:
                    sub_mask.append(True)
            mask.append(sub_mask)
        mask = ak.Array(mask)
        return self.all_jets[mask], self.all_constituent_indices[mask]
