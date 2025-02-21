import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_root(r0, r1, z0, z1, z2, g, max_iterations=10):
    n0, n1 = r0 * z0, r1 * z1
    s0, s1 = z2 - 1, 0 if g < 0 else np.sqrt(n0**2 + n1**2 + z2**2) - 1
    s = 0
    
    for _ in range(max_iterations):
        s = (s0 + s1) / 2
        if s == s0 or s == s1:
            break
        ratio0, ratio1, ratio2 = n0 / (s + r0), n1 / (s + r1), z2 / (s + 1)
        g = ratio0**2 + ratio1**2 + ratio2**2 - 1
        if g > 0:
            s0 = s
        elif g < 0:
            s1 = s
        else:
            break
    
    return s

def distance_point_ellipse(e0, e1, y0, y1):
    eps = 1e-8
    record_sign = [1, 1]
    
    if y0 < 0:
        record_sign[0] = -1
        y0 = -y0
    if y1 < 0:
        record_sign[1] = -1
        y1 = -y1
    
    def get_root(r0, z0, z1, g):
        n0 = r0 * z0
        s0, s1 = z1 - 1, 0 if g < 0 else np.sqrt(n0**2 + z1**2) - 1
        for _ in range(10):
            s = (s0 + s1) / 2
            if s == s0 or s == s1:
                break
            ratio0, ratio1 = n0 / (s + r0), z1 / (s + 1)
            g = ratio0**2 + ratio1**2 - 1
            if g > 0:
                s0 = s
            else:
                s1 = s
        return s
    
    if y1 > eps:
        if y0 > eps:
            z0, z1 = y0 / e0, y1 / e1
            g = z0**2 + z1**2 - 1
            if g != 0:
                r0 = (e0**2) / (e1**2)
                sbar = get_root(r0, z0, z1, g)
                x0, x1 = r0 * y0 / (sbar + r0), y1 / (sbar + 1)
                distance = np.sqrt((x0 - y0)**2 + (x1 - y1)**2)
            else:
                x0, x1, distance = y0, y1, 0
        else:
            x0, x1, distance = 0, e1, abs(y1 - e1)
    else:
        numer0, denom0 = e0 * y0, e0**2 - e1**2
        if numer0 < denom0:
            xde0 = numer0 / denom0
            x0, x1 = e0 * xde0, e1 * np.sqrt(1 - xde0**2)
            distance = np.sqrt((x0 - y0)**2 + x1**2)
        else:
            x0, x1, distance = e0, 0, abs(y0 - e0)
    
    x0 *= record_sign[0]
    x1 *= record_sign[1]
    return distance, x0, x1

def distance_point_ellipsoid(e0, e1, e2, y0, y1, y2):
    eps = 1e-8
    record_sign = [1, 1, 1]
    
    if y0 < 0:
        record_sign[0], y0 = -1, -y0
    if y1 < 0:
        record_sign[1], y1 = -1, -y1
    if y2 < 0:
        record_sign[2], y2 = -1, -y2
    
    if y2 > eps:
        if y1 > eps:
            if y0 > eps:
                z0, z1, z2 = y0 / e0, y1 / e1, y2 / e2
                g = np.sqrt(z0**2 + z1**2 + z2**2) - 1
                
                if g != 0:
                    r0, r1 = e0**2 / e2**2, e1**2 / e2**2
                    sbar = get_root(r0, r1, z0, z1, z2, g)
                    x0, x1, x2 = r0 * y0 / (sbar + r0), r1 * y1 / (sbar + r1), y2 / (sbar + 1)
                    distance = np.sqrt((x0 - y0)**2 + (x1 - y1)**2 + (x2 - y2)**2)
                else:
                    x0, x1, x2, distance = y0, y1, y2, 0
            else:
                x0, distance = 0, distance_point_ellipse(e1, e2, y1, y2)
        else:
            if y0 > 0:
                x1, distance = 0, distance_point_ellipse(e0, e2, y0, y2)
            else:
                x0, x1, x2, distance = 0, 0, e2, abs(y2 - e2)
    else:
        denom0, denom1 = e0**2 - e2**2, e1**2 - e2**2
        numer0, numer1 = e0 * y0, e1 * y1
        computed = False
        
        if numer0 < denom0 and numer1 < denom1:
            xde0, xde1 = numer0 / denom0, numer1 / denom1
            discr = 1 - xde0**2 - xde1**2
            
            if discr > 0:
                x0, x1, x2 = e0 * xde0, e1 * xde1, e2 * np.sqrt(discr)
                distance = np.sqrt((x0 - y0)**2 + (x1 - y1)**2 + x2**2)
                computed = True
        
        if not computed:
            x2, distance = 0, distance_point_ellipse(e0, e1, y0, y1)
    
    return record_sign[0] * x0, record_sign[1] * x1, record_sign[2] * x2, distance

def visualize_ellipsoid(e0, e1, e2, y0, y1, y2, x0, x1, x2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = e0 * np.cos(u) * np.sin(v)
    y = e1 * np.sin(u) * np.sin(v)
    z = e2 * np.cos(v)
    
    ax.plot_surface(x, y, z, color='c', alpha=0.5, edgecolor='k')
    ax.scatter(y0, y1, y2, color='r', label='Query Point')
    ax.scatter(x0, x1, x2, color='b', label='Closest Point')
    ax.plot([y0, x0], [y1, x1], [y2, x2], color='k', linestyle='dashed')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# Example Usage
e0, e1, e2 = 3, 2, 1
y0, y1, y2 = 4, 3, 2
x0, x1, x2, dist = distance_point_ellipsoid(e0, e1, e2, y0, y1, y2)
print(f"Closest Point: ({x0}, {x1}, {x2})")
print(f"Distance: {dist}")
visualize_ellipsoid(e0, e1, e2, y0, y1, y2, x0, x1, x2)
