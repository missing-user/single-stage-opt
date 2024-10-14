import desc.io
import desc.plotting
import desc.grid
import numpy as np
import matplotlib.pyplot as plt

eq_fam = desc.io.load(
    "input.NAS.nv.n4.SS.iota43.Fake-ITER_unperturbed.01_desc_output.h5"
)
eq = eq_fam[-1]
eq_fam2 = desc.io.load("input.NAS.nv.n4.SS.iota43.Fake-ITER.01_output.h5")
eq2 = eq_fam2[-1]

desc.plotting.plot_comparison([eq, eq2])


def manual_Kvc_calculation(eqilibrium, grid):
    MU0 = 4 * np.pi * 1e-6
    computed = eqilibrium.compute(["n_rho", "B"], grid)
    K = np.cross(computed["n_rho"], computed["B"], axis=-1) / MU0
    K_abs = np.linalg.norm(K, axis=-1)
    return K_abs


def get_Kvc_lcfs(eqilibrium, grid):
    computed = eqilibrium.compute(["K_vc"], grid)
    return grid.meshgrid_reshape(computed["K_vc"], "rtz")[0]


lcfs_grid = desc.grid.LinearGrid(theta=100, zeta=100, NFP=eq.NFP)
Kvc = get_Kvc_lcfs(eq, lcfs_grid)
Kvc2 = get_Kvc_lcfs(eq2, lcfs_grid)
norm_of_diff = np.linalg.norm(Kvc - Kvc2, axis=-1)
diff_of_norm = np.linalg.norm(Kvc, axis=-1) - np.linalg.norm(Kvc2, axis=-1)

# Plot grid with K_vc and the differences
plt.figure()
plt.subplot(2, 2, 1)
desc.plotting.plot_2d(eq, "K_vc", ax=plt.gca())
plt.subplot(2, 2, 2)
desc.plotting.plot_2d(eq2, "K_vc", ax=plt.gca())
plt.subplot(2, 2, 3)
plt.contourf(norm_of_diff, cmap="jet", levels=100)
plt.title(f"norm of diff K_vc {np.mean(norm_of_diff):.4e}")
plt.subplot(2, 2, 4)
plt.colorbar()
plt.contourf(diff_of_norm, cmap="jet", levels=100)
plt.title(f"diff of norm K_vc {np.linalg.norm(diff_of_norm):.4e}")
plt.colorbar()
plt.show()
