import desc.io
import desc.plotting
import desc.vmec
import desc.grid
import desc.integrals
import numpy as np
import matplotlib.pyplot as plt

# from desc.compute import rpz2xyz, rtz2xyz

eq_fam = desc.io.load(
    "input.NAS.nv.n4.SS.iota43.Fake-ITER_unperturbed.01_desc_output.h5"
)
eq = eq_fam[-1]

eq_fam2 = desc.io.load("input.NAS.nv.n4.SS.iota43.Fake-ITER.01_output.h5")
eq2 = eq_fam2[-1]


def manual_Kvc_calculation(eqilibrium, grid):
    MU0 = 4 * np.pi * 1e-6
    computed = eqilibrium.compute(["n_rho", "B"], grid)
    K = np.cross(computed["n_rho"], computed["B"], axis=-1) / MU0
    K_abs = np.linalg.norm(K, axis=-1)
    return K_abs


def get_Kvc_lcfs(eqilibrium, grid):
    computed = eqilibrium.compute(["K_vc"], grid)
    return grid.meshgrid_reshape(computed["K_vc"], "rtz")[0]


lcfs_grid = desc.grid.LinearGrid(theta=50, zeta=50, NFP=eq.NFP)
Kvc = get_Kvc_lcfs(eq, lcfs_grid)
Kvc2 = get_Kvc_lcfs(eq2, lcfs_grid)
norm_of_diff = np.linalg.norm(Kvc - Kvc2, axis=-1)
diff_of_norm = np.linalg.norm(Kvc, axis=-1) - np.linalg.norm(Kvc2, axis=-1)


def surf_integral_Kvc(eqilibrium):
    # Integral over the LCFS
    # Exactly integrate polynomials with N,M at the radial position rho=1.0
    quad_grid = desc.grid.LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP)

    return desc.integrals.surface_integrals(
        quad_grid,
        np.linalg.norm(eqilibrium.compute(
            ["K_vc"], quad_grid)["K_vc"], axis=-1),
        expand_out=False,
        surface_label="rho",
    )


Kvc_int = surf_integral_Kvc(eq)
Kvc2_int = surf_integral_Kvc(eq2)
print(Kvc_int)
print(Kvc2_int)

plt.figure()
normal = eq.compute(["n_rho"], lcfs_grid)["n_rho"]
normal = lcfs_grid.meshgrid_reshape(normal, "rtz")[0]
plt.subplot(131)
plt.title("R")
plt.imshow(normal[:, :, 0])
plt.colorbar()
plt.subplot(132)
plt.title("theta")
plt.imshow(normal[:, :, 1])
plt.colorbar()
plt.subplot(133)
plt.title("Z")
plt.imshow(normal[:, :, 2])
plt.colorbar()


plt.figure()
plt.subplot(331)
plt.title("R")
plt.imshow(Kvc[:, :, 0])
plt.colorbar()
plt.subplot(332)
plt.title("theta")
plt.imshow(Kvc[:, :, 1])
plt.colorbar()
plt.subplot(333)
plt.title("Z")
plt.imshow(Kvc[:, :, 2])
plt.colorbar()

plt.subplot(334)
plt.title("R")
plt.imshow(Kvc2[:, :, 0])
plt.colorbar()
plt.subplot(335)
plt.title("theta")
plt.imshow(Kvc2[:, :, 1])
plt.colorbar()
plt.subplot(336)
plt.title("Z")
plt.imshow(Kvc2[:, :, 2])
plt.colorbar()

plt.subplot(337)
plt.title("R")
plt.imshow(Kvc[:, :, 0] - Kvc2[:, :, 0])
plt.colorbar()
plt.subplot(338)
plt.title("theta")
plt.imshow(Kvc[:, :, 1] - Kvc2[:, :, 1])
plt.colorbar()
plt.subplot(339)
plt.title("Z")
plt.imshow(Kvc[:, :, 2] - Kvc2[:, :, 2])
plt.colorbar()

# Plot grid with K_vc and the differences
plt.figure()
plt.subplot(2, 2, 1)
desc.plotting.plot_2d(eq, "K_vc", ax=plt.gca())
plt.subplot(2, 2, 2)
desc.plotting.plot_2d(eq2, "K_vc", ax=plt.gca())
plt.subplot(2, 2, (3, 4))
plt.contourf(diff_of_norm, cmap="jet", levels=100)
plt.title(
    f"Unperturbed surface integral {Kvc_int[0]:.2e}\n"
    + f"Perturbed surface integral {Kvc2_int[0]:.2e}"
)
plt.colorbar()
# plt.subplot(2, 2, 4)
# plt.contourf(diff_of_norm, cmap="jet", levels=100)
# plt.title(f"Perturbed surface integral {Kvc2_int[0]:.2e}")
# plt.colorbar()
plt.show()

desc.vmec.VMECIO.plot_vmec_comparison(
    eq2, "wout_NAS.nv.n4.SS.iota43.Fake-ITER.01_000_000000.nc"
)
desc.plotting.plot_comparison([eq, eq2])

plt.show()
