import desc.io
import desc.plotting
import desc.grid
import numpy as np
import matplotlib.pyplot as plt

eq_fam = desc.io.load("input.NAS.nv.n4.SS.iota43.Fake-ITER.01_output.h5")
eq = eq_fam[-1]
eq_fam2 = desc.io.load(
    "input.NAS.nv.n4.SS.iota43.Fake-ITER.unperturbed.01_output.h5")
eq2 = eq_fam2[-1]

desc.plotting.plot_comparison([eq, eq2])
desc.plotting.plot_2d(eq2, "K_vc")

# desc.plotting.plot_2d(eq, "J")
desc.plotting.plot_2d(eq, "K_vc")
plt.figure()

grid = desc.grid.LinearGrid(theta=100, zeta=100, NFP=eq.NFP)
computed = eq.compute(["n_rho", "B"], grid)
K = np.cross(computed["n_rho"], computed["B"], axis=-1)
K_abs = np.linalg.norm(K, axis=-1)
K_outer = grid.meshgrid_reshape(K_abs, "rtz")[0]
plt.contourf(K_outer)
plt.colorbar()
plt.show()
