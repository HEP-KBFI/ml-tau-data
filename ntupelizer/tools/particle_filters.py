import vector
import awkward as ak
from ntupelizer.tools import general as g


class ParticleFilter:
    def __init__(self, arrays: ak.Array, p_type: str):
        self.p_type = p_type
        self.particles, self.particles_p4 = self.calculate_p4(arrays)
        self.particles, self.particles_p4 = self._filter_particles()

    def calculate_p4(self, arrays: ak.Array):
        particles = ak.Array(
            {
                k.replace(f"{self.p_type}.", ""): arrays[k]
                for k in arrays.fields
                if f"{self.p_type}." in k
            }
        )
        particles_p4 = vector.awk(
            ak.zip(
                {
                    "mass": particles["mass"],
                    "px": particles["momentum.x"],
                    "py": particles["momentum.y"],
                    "pz": particles["momentum.z"],
                }
            )
        )
        return particles, particles_p4

    def _filter_particles(self):
        raise NotImplementedError("Please implement this function for your sub-class")

    @property
    def results(self):
        return self.particles, self.particles_p4


class MCParticleFilter(ParticleFilter):
    def __init__(self, arrays, p_type: str = "MCParticles"):
        super().__init__(arrays=arrays, p_type=p_type)

    def _filter_particles(self):
        """Picks only stable (status=1) particles and removes all neutrinos from the MC particles collection"""
        stable_pythia_mask = self.particles["generatorStatus"] == 1
        neutrino_mask = (
            (abs(self.particles["PDG"]) != 12)
            * (abs(self.particles["PDG"]) != 14)
            * (abs(self.particles["PDG"]) != 16)
        )
        particle_mask = stable_pythia_mask * neutrino_mask
        stable_mc_particles = ak.Array(
            {
                field: self.particles[field][particle_mask]
                for field in self.particles.fields
            }
        )
        stable_mc_p4 = g.reinitialize_p4(self.particles_p4[particle_mask])
        return stable_mc_particles, stable_mc_p4


class RecoParticleFilter(ParticleFilter):
    def __init__(self, arrays, p_type: str = "PandoraPFOs"):
        super().__init__(arrays=arrays, p_type=p_type)

    def _filter_particles(self):
        return self.particles, self.particles_p4
