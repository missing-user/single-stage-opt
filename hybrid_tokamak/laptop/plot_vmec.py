import latexplot
import matplotlib.pyplot as plt
import simsopt.mhd
from simsopt.geo import SurfaceRZFourier
import sys
import numpy as np
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry


def getLgradB(vmec):
    try:
        vmec.run()
        s = [0.25, 1.0]
        ntheta = 32
        nphi = 32
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
        data = vmec_compute_geometry(vmec, s, theta, phi)

        return np.min(data.L_grad_B)
    except Exception as e:
        return np.nan

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

def plot_nml(filename, **kwargs):
    vmec = simsopt.mhd.Vmec(filename)
    lgradb = getLgradB(vmec)
    surf = SurfaceRZFourier.from_vmec_input(filename)
    plot_vmec(surf, **kwargs)
    plt.title(f"{filename}  LgradB={lgradb}")

def plot_vmec(surf, phis=[0, np.pi/2, np.pi], **kwargs):
    labels = ["$\phi = 0$", "$\phi = \pi/(2 n_{fp})$", "$\phi = \pi/n_{fp}$"]
    for phi, label in zip(phis, labels):
        phi = phi / surf.nfp
        cross = surf.cross_section(phi, np.linspace(0, 1, 128, endpoint=True))
        rotated = rotate_to_x_plane(cross, phi)
        plt.plot(rotated[:,0], rotated[:,2], label=label,**kwargs)

    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.legend()

if __name__ == "__main__": 
    vmecs = [simsopt.mhd.Vmec(filename) for filename in sys.argv[1:]]
    # for phi in [0, np.pi/2, np.pi]:
    latexplot.figure()
    latexplot.set_cmap(len(vmecs))
    for vmec, filename in zip(vmecs, sys.argv[1:]):
        phis = [0]
        if len(vmecs) == 1:
            phis = [0, np.pi/2, np.pi]
        plot_nml(filename, phis = phis)
    plt.legend()
    latexplot.savenshow(filename+"_cross_sections")
    for filename in sys.argv[1:]:
        vmec = simsopt.mhd.Vmec(filename)
        # vmec.boundary.plot(show=False)
        vmec.boundary.scale(1)
        vmec.boundary.plot(show=False)
    latexplot.savenshow(filename+"_3d_boundaries")

    