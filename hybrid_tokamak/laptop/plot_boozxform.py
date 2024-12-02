from simsopt import mhd
import matplotlib.pyplot as plt
import numpy as np

import booz_xform
import matplotlib.pyplot as plt
import numpy as np

from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry, vmec_fieldlines

import sys

def boozer_plot(filename):
    vmec = mhd.Vmec(filename)
    boozer = mhd.Boozer(vmec)

    s = [0.25, 1.0]
    ntheta = 4
    nphi = 5
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
    data = vmec_compute_geometry(vmec, s, theta, phi)
    print("data.L_grad_B", data.L_grad_B)

    b1 = boozer.bx
    b1.read_wout(filename)
    b1.compute_surfs = [5, 64]
    b1.run()

    plt.subplot(1, 2, 1)
    booz_xform.surfplot(b1, js=0)
    plt.subplot(1, 2, 2)
    booz_xform.surfplot(b1, js=0, fill=False)
    plt.figure()

    plt.subplot(1, 2, 1)
    booz_xform.surfplot(b1, js=1)
    plt.subplot(1, 2, 2)
    booz_xform.surfplot(b1, js=1, fill=False)
    # plt.figure()
    # booz_xform.symplot(b1)
    plt.show()

if __name__ == "__main__":
    boozer_plot(sys.argv[1])  