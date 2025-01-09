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

def plot_single_vmec(vmec:simsopt.mhd.Vmec, phi=0.0,**kwargs):
    surf = SurfaceRZFourier.from_vmec_input(vmec.input_file)
    phi = phi / surf.nfp
    cross = surf.cross_section(phi, np.linspace(0, 1, 128, endpoint=True))
    rotated = rotate_to_x_plane(cross, phi)

    plt.sca(kwargs.pop("ax", plt.gca()))
    plt.plot(rotated[:,0], rotated[:,2], **kwargs)

    plt.xlabel("R [m]")   
    plt.ylabel("Z [m]")

def compare_crosssections(vmecs, axs=None):
    # for phi in [0, np.pi/2, np.pi]:
    fig=plt.gcf()
    
    # Same behavior as plot_spec
    plot_boundary = False
    if axs is None:
        if len(vmecs) >1:
            fig, axs = plt.subplots(1, 3, figsize=latexplot.get_size(1, (2,3)), sharex=True, sharey=True)
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 1, figsize=latexplot.get_size(1))
            axs = [axs] * 3

    def dof_from_mpol(mpol):
        return mpol*(mpol*2+1)+mpol 

    colors = plt.cm.plasma(np.linspace(0, 1, len(vmecs), endpoint=False))
    colors = np.array(list(reversed(colors)))

    for vmec, c, fi in zip(vmecs, colors, range(len(vmecs))):
        filename = vmec.input_file

        color = c[np.newaxis, :]

        for i, ax in enumerate(axs):  
            label = filename
            label = f"M=N={fi}"
            # label = f"{dof_from_mpol(fi)} DOFs"

            title = f"Fixed-boundary $\phi = {i/(len(axs)-1):.1f} \pi" +"/ n_{fp}$"
            ns = [1]
            if len(vmecs) == 1:
                color = plt.cm.plasma(i/(len(axs)))
                label = title
            if i != 0:
                label = None

            # out.plot_kam_surface(ntheta=128, ns=ns, ax=ax, c=color, label=label, zeta=i*np.pi/(len(axs)-1), linewidth=1)
            plot_single_vmec(vmec, phi=i*np.pi/(len(axs)-1), ax=ax, c=color, label=label, linewidth=1)
            ax.set_title(title)
            if fi == len(vmecs)-1 and plot_boundary:
                # out.plot_kam_surface(ntheta=128, ns=[2], ax=ax, c="black", label="$\mathcal{D}$", zeta=0, linewidth=1)
                plot_single_vmec(vmec, phi=0, c="black", ax=ax, label="$\mathcal{D}$", linewidth=1)
            if i == 0 and fi == len(vmecs)-1:
                if plt.isinteractive():
                    fig.legend(prop={'size': 6}, loc="outside lower right") 
                else:
                    fig.legend(loc="outside lower right")

            plt.axis("auto")
            ax.label_outer()
            if len(vmecs) == 1 and i < len(axs)-1:
                continue    
        plt.axis("auto")
        # plt.legend(filenames)
        latexplot.savenshow(filename+"kam")
        print("Saved", filename+"kam")
  


if __name__ == "__main__": 
    vmecs = [simsopt.mhd.Vmec(filename) for filename in sys.argv[1:]]
    compare_crosssections(vmecs)

    # Plot in one figure
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

    