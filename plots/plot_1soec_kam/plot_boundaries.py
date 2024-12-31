import sys
sys.path.append("/home/missinguser/CSE/single-stage-opt/hybrid_tokamak/laptop/")
import latexplot
latexplot.set_cmap(3)
import matplotlib.pyplot as plt
import numpy as np
import py_spec
import sys
from simsopt import mhd
from simsopt import geo
from spec_rename import SpecRename


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

def plot_surf(boundary):
    for phi, label in zip([0, np.pi/2, np.pi], ["$\phi = 0$", "$\phi = \pi/(2 n_{fp})$", "$\phi = \pi/n_{fp}$"]):
        phi = phi / boundary.nfp
        cross = boundary.cross_section(phi, np.linspace(0, 1, 128, endpoint=True))
        rotated = rotate_to_x_plane(cross, phi)
        plt.plot(rotated[:,0], rotated[:,2], label=label)

    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.legend()

if __name__ == "__main__":
    filenames = sys.argv[1:]
    filename = filenames[0]
    latexplot.figure()

    with SpecRename(filename) as specf:
        spec = mhd.Spec(specf, verbose=False, tolerance=1e-10)
        surf = spec.boundary.to_RZFourier()

    plot_surf(surf)
    compb = spec.computational_boundary.cross_section(0, np.linspace(0, 1, 128, endpoint=True)) 
    plt.plot(compb[:,0], compb[:,2], c="black", label="computational boundary")
    plt.legend()
    latexplot.savenshow(filename.replace(".sp.end", "")+"_boundaries")
    latexplot.figure()
    surf.plot(close=True)
    latexplot.savenshow(filename.replace(".sp.end", "")+"_3d")