import numpy as np
import matplotlib.pyplot as plt
import kerrgeopy as kg
from math import cos, pi

# Number of particles
num_particles = 20
a = 0.999
p = 3
e = 0.4
x = cos(pi/6)
i = 0.1
# Generate orbits with slight variations in initial phases
for i in range(2):
    orbit = kg.StableOrbit(a, p, e, x, initial_phases=(0, 0 + i, 0, 0))
    i = i + 1
    fig, ax = orbit.plot(0,10)
    
plt.show()
