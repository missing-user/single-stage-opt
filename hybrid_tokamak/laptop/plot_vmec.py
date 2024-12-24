import latexplot
import matplotlib.pyplot as plt
import simsopt.mhd
import sys
import numpy as np
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry


def getLgradB(vmec):
    vmec.run()
    s = [0.25, 1.0]
    ntheta = 32
    nphi = 32
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
    data = vmec_compute_geometry(vmec, s, theta, phi)
    return np.min(data.L_grad_B)

def rotate_to_x_plane(crosssec, phi):
    """
    Rotate points in the toroidal cross-section to the x-plane.
    
    Parameters:
        crosssec (numpy.ndarray): Array of shape (npoints, 3) representing points in xyz.
        phi (float): Toroidal angle in radians.
        
    Returns:
        numpy.ndarray: Rotated points in the x-plane of shape (npoints, 3).
    """
    # Rotation matrix to undo toroidal rotation about the z-axis
    rotation_matrix = np.array([
        [np.cos(-phi), -np.sin(-phi), 0],
        [np.sin(-phi),  np.cos(-phi), 0],
        [0,             0,            1]
    ])
    
    # Apply rotation to each point
    rotated_points = crosssec @ rotation_matrix.T
    return rotated_points

def plot_vmec(filename, phis=[0, np.pi/2, np.pi]):
    vmec = simsopt.mhd.Vmec(filename)
    for phi in phis:
        cross = vmec.boundary.cross_section(phi)
        rotated = rotate_to_x_plane(cross, phi)
        plt.plot(rotated[:,0], rotated[:,2], label=f"{filename}  LgradB={getLgradB(vmec)}")

if __name__ == "__main__": 
    vmecs = [simsopt.mhd.Vmec(filename) for filename in sys.argv[1:]]
    # for phi in [0, np.pi/2, np.pi]:
    latexplot.figure()
    latexplot.set_cmap(len(vmecs))
    for vmec, filename in zip(vmecs, sys.argv[1:]):
        phis = [0]
        if len(vmecs) == 1:
            phis = [0, np.pi/2, np.pi]
        plot_vmec(filename, phis)
    plt.legend()
    latexplot.savenshow("vmec_cross_sections")
    for filename in sys.argv[1:]:
        vmec = simsopt.mhd.Vmec(filename)
        # vmec.boundary.plot(show=False)
        vmec.boundary.scale(1)
        vmec.boundary.plot(show=False)
    latexplot.savenshow("vmec_3d_boundaries")

    