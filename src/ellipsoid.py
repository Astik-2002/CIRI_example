import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

# Noise parameters
robot_size = 0.2
k1 = 0.04*np.sqrt(11.35) + robot_size # Lateral noise in x [m]
k2 = 0.04*np.sqrt(11.35) + robot_size # Lateral noise in y [m]
a, b, c = 0.001063, 0.0007278, 0.003949  # Depth noise coefficients

# Function to minimize (C^-1(p-d) . (p-s))
def objective(p, C_inv, d, s):
    return np.dot(C_inv @ (p - d), (p - s))

# Constraint: (p-d)^T C^-1 (p-d) = 1
def ellipsoid_constraint(p, C_inv, d):
    return np.dot((p - d).T, C_inv @ (p - d)) - 1

# Main solver function
def solve(C, d, s):
    C_inv = np.linalg.inv(C)
    
    # Initial guess (p close to d)
    p0 = d + np.random.rand(3)
    
    # Constraints definition
    constraints = ({'type': 'eq', 'fun': ellipsoid_constraint, 'args': (C_inv, d)})
    
    # Minimize the objective function with constraint
    result = minimize(objective, p0, args=(C_inv, d, s), constraints=constraints)
    
    if result.success:
        p = result.x
        return p
    else:
        raise RuntimeError("Optimization failed")

def from_two_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    print("v1:", v1, "shape:", v1.shape)
    print("v2:", v2, "shape:", v2.shape)

    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)
    s = np.sqrt((1 + dot_prod) * 2)
    q = np.array([s * 0.5, cross_prod[0] / s, cross_prod[1] / s, cross_prod[2] / s])
    return Rotation.from_quat(q).as_matrix()

# Function to calculate depth noise
def s_z(z):
    return (a + b * z + c * z**2)*np.sqrt(11.35) + robot_size

def s_z_kinect(z):
    return 0.0012 + 0.0019*(z-0.4)**2

# Covariance matrix
def covariance_matrix(x, y, z):
    sz = s_z(z)
    k1a = k1/10*z
    k2a = k2/10*z
    Sigma = np.array([
        [k1**2 + (x**2 * sz**2) / z**2, (x * y * sz**2) / z**2, (x * sz**2) / z],
        [(x * y * sz**2) / z**2, k2**2 + (y**2 * sz**2) / z**2, (y * sz**2) / z],
        [(x * sz**2) / z, (y * sz**2) / z, sz**2]
    ])
    return Sigma

def covariance_matrix_kinect(x, y, z):
    sz = s_z_kinect(z)*np.sqrt(11.35) + robot_size
    c1 = x/z
    c2 = y/z
    sl = 0.8/540*np.sqrt(11.35) + robot_size
    Sigma = np.array([
        [sl**2, 0, c1*sz**2],
        [0, sl**2, c2*sz**2],
        [c1*sz**2, c2*sz**2, sz**2]
    ])
    return Sigma

def plot_sphere(ax, center, radius, color='green', alpha=0.2):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    
    # Shift sphere to center point
    x += center[0]
    y += center[1]
    z += center[2]
    
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)

def compute_tangent_plane_outside_pt(mean, cov, pass_pt, seed_pt):
    cov_inv = np.linalg.inv(cov)
    pass_pt_E = cov_inv @ (pass_pt - mean)
    seed_pt_E = cov_inv @ (seed_pt - mean)
    r = 1
    c = np.array([0, 0, 0])
    pc = pass_pt_E - c
    sc = seed_pt_E - c
    n = np.cross(pc, sc)
    n = n / np.linalg.norm(n)
    R_matrix = from_two_vectors(n, np.array([0, 0, 1]))
    p = R_matrix @ pc
    C = R_matrix @ sc
    r2 = r*r
    p1p2n = np.sqrt(p[0]**2 + p[1]**2)
    d = np.sqrt(p1p2n - r2)
    rp1p2n = r / p1p2n
    q11 = rp1p2n * (p[0] * r - p[1] * d)
    q12 = rp1p2n * (p[1] * r + p[0] * d)

    q21 = rp1p2n * (p[0] * r + p[1] * d)
    q22 = rp1p2n * (p[1] * r - p[0] * d)
    Q = np.array([0, 0, 0])
    if  q11 * C[0] + q21 * C[1] < 0:
        Q[0] = q12
        Q[1] = q22
        
    else:
        Q[0] = q11
        Q[1] = q21
    
    Q[2] = 0
    outer_plane = np.array([0, 0, 0, 0])
    outer_plane[:3] = R_matrix.T @ Q
    Q = outer_plane[:3] + c
    outer_plane[3] = -Q.dot(outer_plane[:3])
    print('outer plane: ', outer_plane)
    return outer_plane 

    outer_plane[:3] = outer_plane[:3].T @ cov
    outer_plane[3] = outer_plane[3] - outer_plane[:3].dot(mean)

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

def compute_tangent_plane(point, mean, cov):
    cov_inv = np.linalg.inv(cov)
    normal = cov_inv @ (point - mean)
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    return normal

# Function to plot the tangent plane
def plot_tangent_plane(ax, point, normal, size=10.0, color='orange', alpha=0.3):
    if normal.shape[0] == 4:
        d = normal[3]
        normal = normal[:3]
    
    d = -np.dot(normal, point) if len(normal) == 3 else d
    
    xx, yy = np.meshgrid(np.linspace(point[0] - size, point[0] + size, 20),
                         np.linspace(point[1] - size, point[1] + size, 20))
    
    if np.abs(normal[2]) > 1e-6:  # Regular plane
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
    else:  # Vertical plane (normal mostly in XY plane)
        zz = np.zeros_like(xx) + point[2]

    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)
   

# Function to generate a random point on the surface of the ellipsoid
def random_point_on_ellipsoid(mean, cov):
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(S)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = radii[0] * np.sin(phi) * np.cos(theta)
    y = radii[1] * np.sin(phi) * np.sin(theta)
    z = radii[2] * np.cos(phi)
    point = np.array([x, y, z])
    return mean + U @ point

# Draw ellipsoid
def plot_ellipsoid(ax, mean, cov, color='blue', alpha=0.2):
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(S)
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 10)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x[i])):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            point = U @ np.diag(radii) @ point
            x[i, j], y[i, j], z[i, j] = point + mean
    print("U mat: ",U)
    print("radii: ",radii)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)



# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_xlim([0, 16])
ax.set_ylim([-8, 8])
ax.set_zlim([-8, 8])

# Generate random pointcloud and plot ellipsoids
np.random.seed(42)
num_points = 100
hfov = np.radians(60)  # Horizontal FOV in radians
vfov = np.radians(60)  # Vertical FOV in radians

xs = np.random.uniform(1, 7, num_points)
# ys = np.random.uniform(-2, 2, num_points)
# zs = np.random.uniform(-2, 2, num_points)
ys = np.random.uniform(-np.tan(hfov/2) * xs, np.tan(hfov/2) * xs, num_points)  # Horizontal range (Y axis)
zs = np.random.uniform(-np.tan(vfov/2) * xs, np.tan(vfov/2) * xs, num_points)  # Vertical range (Z axis)

R_tf = np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])
R_tf_inv = np.linalg.inv(R_tf)

mean = np.array([4.0, 0.0, 3.0])
mean2 = R_tf_inv @ mean
cov = covariance_matrix(mean2[0],mean2[1],mean2[2])
cov = R_tf @ cov @ R_tf.T

cov2 = covariance_matrix_kinect(mean2[0],mean2[1],mean2[2])
cov2 = R_tf @ cov2 @ R_tf.T

plot_sphere(ax, mean, robot_size) # green spheres of robot radius

# cov2 = covariance_matrix(mean2[0], mean2[1], mean2[2]) # old covariance from D435 paper
# cov2 = R_tf @ cov2 @ R_tf.T
plot_ellipsoid(ax, mean, cov, color='blue', alpha = 0.2)
# plot_ellipsoid(ax, mean, cov2, color='red', alpha = 0.2)
for i in range(2):
    random_point = random_point_on_ellipsoid(mean, cov)
    normal = compute_tangent_plane(random_point, mean, cov)
    plot_tangent_plane(ax, random_point, normal, size=1.0, color='orange', alpha=0.3)
    ax.scatter(random_point[0], random_point[1], random_point[2], color='purple', label='Random Point')

pass_point = np.array([5.0, 3.0, 6.0])
seed_point = np.array([0.0, 1.0, 0.0])

e_p = solve(cov, mean, pass_point)
print("checking e_p lies on cov: ", (e_p - mean).T @ np.linalg.inv(cov) @ (e_p - mean))

tangent = compute_tangent_plane(pass_point, mean, cov)
# cov_inv = np.linalg.inv(cov)
# pass_point_E = cov_inv @ (pass_point - mean)
# seed_point_E = cov_inv @ (seed_point - mean)
# o = np.array([0.0, 0.0, 0.0])
# tangent_e = find_tangent_plane_of_sphere(o, 1, pass_point_E, seed_point_E) # compute_tangent_plane_outside_pt(mean, cov, pass_point, seed_point)
# # Eigen::Vector4d plane_w;
# # plane_w.head(3) = plane_e.head(3).transpose() * C_inv_;
# # plane_w(3) = plane_e(3) - plane_w.head(3).dot(d_);
# tangent = np.array([0.0, 0.0, 0.0, 0.0])
# tangent[:3] = tangent_e[:3].T @ cov_inv
# tangent[3] = tangent_e[3] -tangent[:3].dot(mean) 

# plot_tangent_plane(ax, pass_point, tangent[:3], size=10.0, color='red', alpha=0.3)
# plot_ellipsoid(ax, np.array([5.0, 3.0, 3.0]), np.array([[1, 0, 0],
#                      [0, 1, 0],
#                      [0, 0, 1]]), color='blue', alpha = 0.2)
# ax.scatter(pass_point[0], pass_point[1], pass_point[2], color='black', label='Random Point')
# ax.scatter(e_p[0], e_p[1], e_p[2], color='black', label='Random Point')

ax.scatter(mean[0], mean[1], mean[2], color='black')
# plt.title('3D Pointcloud Noise Visualization with Tangent Plane')
# plt.legend()
# plt.show()

# ax.scatter(mean[0], mean[1], mean[2], color='black')

# print("shape mat: ",cov)
# print("center: ",mean)
plt.title('3D Pointcloud Noise Visualization')
plt.show()