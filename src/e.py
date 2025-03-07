import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def plot_sphere(ax, center=np.array([0, 0, 0]), radius=1, color='b', alpha=0.3):
    """ Plot a sphere with a given center and radius """
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='k')

def plot_plane(ax, normal, point, size=2, color='g', alpha=0.5):
    """ Plot a plane given by a normal vector and a point on the plane """
    normal = normal / np.linalg.norm(normal)  # Normalize normal vector
    d = -np.dot(normal, point)
    
    # Check if the plane is vertical (normal[2] ≈ 0)
    if abs(normal[2]) < 1e-6:
        xx, zz = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
        yy = (-normal[0] * xx - normal[2] * zz - d) / normal[1]
    else:
        xx, yy = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
    
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)

def find_tangent_plane_of_sphere(center, r, pass_point, seed_p):
    seed = seed_p.copy()
    dif = pass_point - seed_p
    if np.linalg.norm(dif) < 1e-3:
        if np.linalg.norm((pass_point - center)[:2]) > 1e-3:
            v1 = (pass_point - center) / np.linalg.norm(pass_point - center)
            v1[2] = 0
            seed += 0.01 * np.cross(v1, np.array([0, 0, 1])) / np.linalg.norm(np.cross(v1, np.array([0, 0, 1])))
        else:
            seed += 0.01 * np.cross(pass_point - center, np.array([1, 0, 0])) / np.linalg.norm(np.cross(pass_point - center, np.array([1, 0, 0])))

    P = pass_point - center
    norm_ = np.cross(pass_point - center, seed - center)
    norm_ /= np.linalg.norm(norm_)
    R = Rotation.from_rotvec(np.cross(norm_, np.array([0, 0, 1]))).as_matrix()

    P = R @ P
    C = R @ (seed - center)

    r2 = r * r
    p1p2n = np.sum(P[:2] ** 2)
    d = np.sqrt(p1p2n - r2)
    rp1p2n = r / p1p2n

    q11 = rp1p2n * (P[0] * r - P[1] * d)
    q21 = rp1p2n * (P[1] * r + P[0] * d)
    q12 = rp1p2n * (P[0] * r + P[1] * d)
    q22 = rp1p2n * (P[1] * r - P[0] * d)

    if q11 * C[0] + q21 * C[1] < 0:
        Q = np.array([q12, q22, 0])
    else:
        Q = np.array([q11, q21, 0])

    outer_plane = np.zeros(4)
    outer_plane[:3] = R.T @ Q
    Q_world = outer_plane[:3] + center
    outer_plane[3] = -np.dot(Q_world, outer_plane[:3])

    
    if np.dot(outer_plane[:3], seed) + outer_plane[3] > 1e-6:
        outer_plane *= -1

    return outer_plane

# Given shape matrix C and mean d
C = np.array([[2, 0.5, 0], [0.5, 1, 0], [0, 0, 1.5]])  # Full shape matrix
d = np.array([1, 2, 3])  # Mean of the ellipsoid
p = np.array([4, 6, 3])  # External point
s = np.array([0.0, 0.0, 0.0])
# Step 1: Transform point to unit sphere frame
C_inv = np.linalg.inv(C)
p_transformed = C_inv @ (p - d)  # Convert p to unit sphere frame
s_transformed = C_inv @ (s - d)
# # Step 2: Compute tangent plane in unit sphere frame
# normal_sphere = p_transformed / np.linalg.norm(p_transformed)  # Normal to unit sphere

# # Step 3: Transform normal back to original frame
# normal_ellipsoid = np.linalg.inv(C.T) @ normal_sphere  # Use C^{-T}

normal_sphere = find_tangent_plane_of_sphere(s, 1, p_transformed, s_transformed)
normal_sphere_p = normal_sphere[:3]
normal_ellipsoid_p = np.linalg.inv(C.T) @ normal_sphere_p
# Visualization
fig = plt.figure(figsize=(12, 6))

# --- Left plot: Unit sphere ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Unit Sphere and Tangent Plane")
plot_sphere(ax1, radius=1, color='b', alpha=0.3)
plot_plane(ax1, normal_sphere_p, p_transformed, size=2, color='g', alpha=0.5)
ax1.scatter(*p_transformed, color='r', s=50, label="External Point (Transformed)")
ax1.legend()
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_zlim(-2, 2)

# --- Right plot: Ellipsoid ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Ellipsoid and Tangent Plane")

# Generate and transform sphere surface to ellipsoid
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

ellipsoid_points = np.array([x.flatten(), y.flatten(), z.flatten()])
ellipsoid_transformed = (C @ ellipsoid_points).T + d  # Transform and shift
X_ellipsoid = ellipsoid_transformed[:, 0].reshape(x.shape)
Y_ellipsoid = ellipsoid_transformed[:, 1].reshape(y.shape)
Z_ellipsoid = ellipsoid_transformed[:, 2].reshape(z.shape)

ax2.plot_surface(X_ellipsoid, Y_ellipsoid, Z_ellipsoid, color='b', alpha=0.3, edgecolor='k')
plot_plane(ax2, normal_ellipsoid_p, p, size=5, color='g', alpha=0.5)
ax2.scatter(*p, color='r', s=50, label="External Point")
ax2.legend()
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_zlim(-5, 5)

plt.show()
