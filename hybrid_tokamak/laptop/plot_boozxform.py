from simsopt import mhd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry

import sys
from spec_rename import SpecRename

def getLgradB(vmec:mhd.Vmec):
    vmec.run()
    s = [0.25, 1.0]
    ntheta = 32
    nphi = 32
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
    data = vmec_compute_geometry(vmec, s, theta, phi)
    return np.min(data.L_grad_B)

def pbooz(vmec, sarr, nrows=2, **kwargs): 
    import booz_xform
    from matplotlib import cm
    vmec.run()

    boozer = mhd.Boozer(vmec,**kwargs)
    boozer.register(sarr)
    boozer.run()

    if "cmap" not in kwargs:
        kwargs["cmap"] = cm.plasma
    if "fill" not in kwargs:
        kwargs["fill"] = False

    nrows = 2
    cols = int(np.ceil(len(sarr)/nrows))
    for i, js in enumerate(sarr):
        plt.subplot(nrows, cols, i+1)
        booz_xform.surfplot(boozer.bx, i, **kwargs)

    # plt.suptitle(vmec.filename)

if __name__ == "__main__":
    for filename in sys.argv[1:]: 
        if filename.startswith("input."): 
            vmec = mhd.Vmec(filename)
        else:
            vmec = mhd.Vmec("hybrid_tokamak/laptop/input.rot_ellipse")
            with SpecRename(filename) as specf:
                spec = mhd.Spec(specf)
                vmec.boundary = spec.boundary.copy()
        plt.figure()
        pbooz(vmec, np.array([0.25, 0.5, 0.751, 1.0]))
        plt.suptitle(filename+f"\nLgradB={getLgradB(vmec):.3f}")
    plt.show()