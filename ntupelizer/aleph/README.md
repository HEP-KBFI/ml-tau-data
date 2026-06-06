# Issues found with ALEPH data:

- some constituents have a p_T of -1e3. Need to clean recoParticles.
- some jets have 0 particles -> assigned particles to jets based on angle.
- As some jets have 0 associated particles, then the nparticles property needs to be taken from the angle-associated particles.
- For some constituents the eta value is infinite / NaN
- some jets have infinite eta value (from the EDM4HEP files.)