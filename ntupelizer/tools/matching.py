import numba
import vector
import numpy as np
import awkward as ak
from ntupelizer.tools import general as g


@numba.njit
def deltar(eta1, phi1, eta2, phi2):
    deta = np.abs(eta1 - eta2)
    dphi = deltaphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


@numba.njit
def deltaphi(phi1, phi2):
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


@numba.njit
def dr_matching(collection1: ak.Array, collection2: ak.Array, delta_r: float = 0.4):
    iev = len(collection1)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    for ev in range(iev):
        j1 = collection1[ev]
        j2 = collection2[ev]

        jet_inds_1 = []
        jet_inds_2 = []
        for ij1 in range(len(j1)):
            if j1[ij1].energy == 0:
                continue
            drs = np.zeros(len(j2), dtype=np.float64)
            for ij2 in range(len(j2)):
                if j2[ij2].energy == 0:
                    continue
                eta1 = j1.eta[ij1]
                eta2 = j2.eta[ij2]
                phi1 = j1.phi[ij1]
                phi2 = j2.phi[ij2]

                # Workaround for https://github.com/scikit-hep/vector/issues/303
                # dr = j1[ij1].deltaR(j2[ij2])
                dr = deltar(eta1, phi1, eta2, phi2)
                drs[ij2] = dr
            if len(drs) > 0:
                min_idx_dr = np.argmin(drs)
                if drs[min_idx_dr] < delta_r:
                    jet_inds_1.append(ij1)
                    jet_inds_2.append(min_idx_dr)
        jet_inds_1_ev.append(jet_inds_1)
        jet_inds_2_ev.append(jet_inds_2)
    return jet_inds_1_ev, jet_inds_2_ev


class JetMatcher:
    def __init__(
        self,
        reco_jets: ak.Array,
        gen_jets: ak.Array,
        reco_jet_constituent_indices: ak.Array,
        delta_r: float,
    ):
        self.delta_r = delta_r
        reco_jets = g.reinitialize_p4(reco_jets)
        gen_jets = g.reinitialize_p4(gen_jets)
        self.reco_jets, self.gen_jets, self.reco_jet_constituent_indices = (
            self.match_jets(reco_jets, gen_jets, reco_jet_constituent_indices)
        )

    def match_jets(self, reco_jets, gen_jets, reco_jet_constituent_indices):
        reco_indices, gen_indices = dr_matching(
            collection1=reco_jets, collection2=gen_jets, delta_r=self.delta_r
        )
        reco_jet_constituent_indices = ak.from_iter(
            [reco_jet_constituent_indices[i][idx] for i, idx in enumerate(reco_indices)]
        )
        reco_jets = ak.from_iter(
            [reco_jets[i][idx] for i, idx in enumerate(reco_indices)]
        )
        gen_jets = ak.from_iter([gen_jets[i][idx] for i, idx in enumerate(gen_indices)])
        return (
            g.reinitialize_p4(reco_jets),
            g.reinitialize_p4(gen_jets),
            reco_jet_constituent_indices,
        )

    @property
    def results(self):
        return self.reco_jets, self.gen_jets, self.reco_jet_constituent_indices


def get_jet_constituent_property(
    property_: str, constituent_idx: int, num_ptcls_per_jet: int
):
    reco_property_flat = property_[ak.flatten(constituent_idx, axis=-1)]
    return ak.from_iter(
        [
            ak.unflatten(reco_property_flat[i], num_ptcls_per_jet[i], axis=-1)
            for i in range(len(num_ptcls_per_jet))
        ]
    )
