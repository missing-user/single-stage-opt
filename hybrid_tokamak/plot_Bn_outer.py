import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf
import sys

if len(sys.argv) != 2:
    print("Error! You must specify 1 argument: the bdistrib_out.XXX.nc file.")
    # exit(1)
    filename = "bdistrib_out.hybrid_tokamak.nc"
else:
    filename = sys.argv[1]

with netcdf.netcdf_file(filename, "r", mmap=False) as f:
    nfp = f.variables["nfp"][()]
    transferMatrix = f.variables["transferMatrix"][()]

    try:
        # Fourier components of B.n on the plasma surface
        Bnormal_from_const_v_coils = f.variables["Bnormal_from_const_v_coils"][(
        )]
        # 2D array of B.n on the plasma surface (iFFT of Bnormal_from_const_v_coils)
        Bnormal_from_const_v_coils_uv = f.variables["Bnormal_from_const_v_coils_uv"][(
        )]
        Bnormal_from_const_v_coils_inductance = f.variables[
            "Bnormal_from_const_v_coils_inductance"
        ][()]
        Bnormal_from_const_v_coils_transfer = f.variables[
            "Bnormal_from_const_v_coils_transfer"
        ][()]
        const_v_exists = True
    except:
        const_v_exists = False
    try:
        # (num_basis_functions_plasma, nu_nv_plasma)
        basis_functions_plasma = f.variables["basis_functions_plasma"][()]
        nu_plasma = f.variables["nu_plasma"][()]
        nv_plasma = f.variables["nv_plasma"][()]
        norm_normal_plasma = f.variables["norm_normal_plasma"][()]
        # (num_basis_functions_outer, nu_nv_outer)
        basis_functions_outer = f.variables["basis_functions_outer"][()]
        nu_outer = f.variables["nu_outer"][()]
        nv_outer = f.variables["nv_outer"][()]
        norm_normal_outer = f.variables["norm_normal_outer"][()]

    except:
        print(
            "Could not load basis functions, decrease to save_level=0 in bdistrib, or they won't get written."
        )
print(norm_normal_plasma.shape)
print(
    transferMatrix.shape,
    Bnormal_from_const_v_coils.shape,
    basis_functions_outer.shape,
    basis_functions_plasma.shape,
)


plt.subplot(121)
plt.contourf(Bnormal_from_const_v_coils_uv.T, 100)
plt.xlabel("v", fontsize="x-small")
plt.ylabel("u", fontsize="x-small")
# plt.imshow(,interpolation='none')
# plt.gca().xaxis.tick_top()
plt.colorbar()
plt.title("B field due to constant-v coils\n(component normal to plasma surface)")

plt.subplot(122)

Bnormal_plasma_basis_functions_uv = (
    ((basis_functions_plasma.T @ Bnormal_from_const_v_coils))
    .reshape(nu_plasma, nv_plasma)
    .T
)

plt.contourf(Bnormal_plasma_basis_functions_uv, 100)
plt.title("B field due to constant-v coils\n(from basis decomposition)")
plt.colorbar()

plt.figure()
plt.plot(Bnormal_from_const_v_coils)
Bnormal_from_basis_change = (
    basis_functions_plasma @ Bnormal_from_const_v_coils_uv.flatten() * 1e-3
)
plt.plot(Bnormal_from_basis_change)
plt.title("Check if basis swap changes the result")

plt.figure()
plt.subplot(121)
# (Boozer eq. 44) \begin{equation} \vec{\Phi}_x=\,{\buildrel{\leftrightarrow}\over{T}}\cdot\vec{\Phi}_c. \end{equation}
# Phi_plasma = T Phi_coils
Bnormal_on_outer = np.linalg.lstsq(
    transferMatrix.T, Bnormal_from_const_v_coils, rcond=None
)[0]
plt.plot(Bnormal_on_outer)

# Bnormal_on_outer = transferMatrix @ Bnormal_from_const_v_coils
Bnormal_on_outer = np.linalg.lstsq(
    transferMatrix.T, Bnormal_from_const_v_coils, rcond=None
)[0]
Bnormal_outer_basis_functions_uv = np.reshape(
    (basis_functions_outer.T @ Bnormal_on_outer), (nu_outer, nv_outer)
).T
plt.subplot(122)
plt.contourf(Bnormal_outer_basis_functions_uv, 100)
plt.colorbar()
plt.title("$B_{outer} \\cdot n$")


plt.matshow(transferMatrix)
plt.title(f"Transfer Matrix (cond: {np.linalg.cond(transferMatrix):.4e})")
plt.colorbar()

# Why does the basis transform smooth out B.n so much???
plt.matshow(
    (
        basis_functions_plasma.T
        @ basis_functions_plasma
        @ Bnormal_from_const_v_coils_uv.flatten()
    ).reshape(nu_plasma, nv_plasma)
)
plt.colorbar()
plt.title("$U^T U B \\cdot n$")

plt.show()
