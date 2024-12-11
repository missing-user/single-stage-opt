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

if __name__ == "__main__": 
    vmecs = [simsopt.mhd.Vmec(filename) for filename in sys.argv[1:]]
    # for phi in [0, np.pi/2, np.pi]:
    latexplot.figure(1)
    for vmec, filename in zip(vmecs, sys.argv[1:]):
        cross = vmec.boundary.cross_section(0)
        plt.plot(cross[:,0], cross[:,2], label=f"{filename}  LgradB={getLgradB(vmec)}")
    plt.legend()
    latexplot.savenshow("vmec_cross_sections")
    for filename in sys.argv[1:]:
        vmec = simsopt.mhd.Vmec(filename)
        # vmec.boundary.plot(show=False)
        vmec.boundary.scale(1)
        vmec.boundary.plot(show=False)
    latexplot.savenshow("vmec_3d_boundaries")

    