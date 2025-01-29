import numpy as np
import matplotlib.pyplot as plt
import kerrgeopy as kg
from math import cos, pi

# Number of particles
num_particles = 20

# Generate orbits with slight variations in initial phases
for i in range(num_particles):
    # Slight variation in initial phases
    initial_phases = {
        'psi_r': 1e-2 * i + 0.0,
        'chi': 0.0,
        'phi': 0.0
    }
    # Create the stable orbit
    orbit = kg.StableOrbit(0.99, 3, 0.4, cos(pi/7), initial_phases=initial_phases)

fig, ax = orbit.plot(0,10)