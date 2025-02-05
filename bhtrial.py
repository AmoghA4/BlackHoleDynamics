import numpy as np
import matplotlib.pyplot as plt
import kerrgeopy as kg
from math import cos, pi

# Create a single figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Black hole and Orbit Parameters
a = 0.998
p = 3
e = 0.4
x = cos(pi / 6)
M = 1e12

# Initial Position Values
t_initial = 0
r_iniital = 5
theta_initial = pi / 4
phi_initial = pi / 2

# Initial Velocity Values
vt_initial = 0
vr_initial = -0.5
vtheta_initial = 0
vphi_initial = 0

# Perturbation Terms
delta_r = 0.1
delta_theta = 0.05
number_of_particles = 5

# Time Evolution
lambda_1 = 20
lambda_2 = 45
samples = 1000

# Function to Solve for Fourth Component of Velocity Given Three
def compute_fourth_velocity_component(metric, u_initial):
    """
    Compute the fourth component of the 4-velocity given the metric tensor and the first three components.

    Parameters:
    metric (np.ndarray): 4x4 matrix representing the metric tensor.
    u_initial (np.ndarray): 3-component array representing the first three components of the 4-velocity.

    Returns:
    float: The fourth component of the 4-velocity.
    """
    u = np.zeros(4)  # Full 4-velocity vector
    u[1:] = u_initial  # Assign given components

    # Solve for u^0 using the norm condition: g_{μν} u^μ u^ν = -1
    norm = (
        metric[0, 0] * u[0] ** 2
        + 2 * np.dot(metric[0, 1:], u[1:]) * u[0]
        + np.dot(u[1:], metric[1:, 1:].dot(u[1:]))
    )

    # Solve for u[0] (assuming a physically meaningful positive root)
    a = metric[0, 0]
    b = 2 * np.dot(metric[0, 1:], u[1:])
    c = np.dot(u[1:], metric[1:, 1:].dot(u[1:])) + 1  # Enforce norm condition

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real solution for u^0, check the inputs.")

    u0_1 = (-b + np.sqrt(discriminant)) / (2 * a)
    u0_2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # Choose the physically relevant root (typically positive for future-directed velocities)
    u[0] = u0_1 if u0_1 > 0 else u0_2

    return u[0]


# Plot the Orbits Given The Initial Conditions
for orbit_num in range(number_of_particles):

    spacetime = kg.KerrSpacetime(a, M)
    z = spacetime.metric(
        t_initial, r_iniital + delta_r, theta_initial + delta_theta, phi_initial
    )
    g_metric = z
    u_initial = np.array([vr_initial, vtheta_initial, vphi_initial])
    u0 = compute_fourth_velocity_component(g_metric, u_initial)

    orbit = kg.Orbit(
        a,
        initial_position=(
            t_initial,
            r_iniital + delta_r,
            theta_initial + delta_theta,
            phi_initial,
        ),
        initial_velocity=(u0, vr_initial, vtheta_initial, vphi_initial),
    )
    # trajectory functions
    t, r, theta, phi = orbit.trajectory()
    delta_r = delta_r + 0.05
    delta_theta = delta_theta = 0.01
    # points for plotting
    lambda_vals = np.linspace(0, 10, 1000)

    # spherical to Cartesian coords
    x_vals = r(lambda_vals) * np.sin(theta(lambda_vals)) * np.cos(phi(lambda_vals))
    y_vals = r(lambda_vals) * np.sin(theta(lambda_vals)) * np.sin(phi(lambda_vals))
    z_vals = r(lambda_vals) * np.cos(theta(lambda_vals))

    # Ploting on single set of axes
    ax.plot(x_vals, y_vals, z_vals)

spacetime = kg.KerrSpacetime(a, M)
z = spacetime.metric(
    t_initial, r_iniital + delta_r, theta_initial + delta_theta, phi_initial
)
g_metric = z
u_initial = np.array([vr_initial, vtheta_initial, vphi_initial])
u0 = compute_fourth_velocity_component(g_metric, u_initial)

# Extract Orbital Parameters and Convert Them To Cartesian
orbit = kg.Orbit(
    a,
    initial_position=(
        t_initial,
        r_iniital + delta_r,
        theta_initial + delta_theta,
        phi_initial,
    ),
    initial_velocity=(u0, vr_initial, vtheta_initial, vphi_initial),
)
fig, ax = orbit.plot(lambda_1,lambda_2,tau=5,elevation=30,azimuth=60)

# Plot Parameters
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=30, azim=-60)
plt.show()