from simsopt.mhd import Spec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
from simsopt import geo
from simsopt.util import MpiPartition
import matplotlib.pyplot as plt
import numpy as np

mpi = MpiPartition(ngroups=4)


equil = Spec("hybrid_tokamak/laptop/hybrid_tokamak_vacuum.sp")
assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"

Bn = equil.normal_field  # This is our new fancy-pants degree of freedom :)

msurf = geo.SurfaceRZFourier.from_vmec_input("hybrid_tokamak/input.hybrid_tokamak")
msurf.change_resolution(msurf.mpol, 0)
msurf.scale(1.5)
msurf.change_resolution(equil.lib.inputlist.mpol, equil.lib.inputlist.ntor)

equil.normal_field.surface = msurf
# m = equil.array_translator(msurf.rc).as_spec.shape[0]
# n = equil.array_translator(msurf.rc).as_spec.shape[1]
# xm, xn = np.meshgrid(np.arange(m), np.arange(n))
# initial_guess = np.vstack(
#     (
#         equil.array_translator(msurf.rc).as_spec.flatten(),
#         equil.array_translator(msurf.zs).as_spec.flatten(),
#         equil.array_translator(msurf.rs).as_spec.flatten(),
#         equil.array_translator(msurf.zc).as_spec.flatten(),
#     )
# ).T

#     mn = np.column_stack((xm.flatten(), xn.flatten()))
#     initial_guess = np.hstack((mn, initial_guess))
#     print("initial_guess.shape", initial_guess.shape)
#     fmt = ["%d", "%d"] + ["%f"] * 4 * equil.lib.inputlist.nvol
#     np.savetxt(f, initial_guess, fmt=fmt)
# msurf.m

with open("hybrid_tokamak/laptop/hybrid_tokamak_vacuum.sp", "a") as f:
    f.write("\n")
    surface = equil.boundary
    for m in range(surface.mpol + 1):
        nmin = -surface.ntor
        if m == 0:
            nmin = 0
        for n in range(nmin, surface.ntor + 1):
            line = f"   {m:5d}    {n:5d} "

            rc = surface.get_rc(m, n)
            zs = surface.get_zs(m, n)
            rs = 0.0
            zc = 0.0
            if not surface.stellsym:
                rs = surface.get_rs(m, n)
                zc = surface.get_zc(m, n)

            # Interpolate the initial guesses for the Fourier coefficients linearly from the outermost wall until the axis
            for l in range(equil.lib.inputlist.nvol):
                scale = 1.0 - l / equil.lib.inputlist.nvol
                line += f" {rc*scale:22.15E} {zs*scale:22.15E} {rs*scale:22.15E} {zc*scale:22.15E}"
            f.write(line + "\n")

print(equil.dof_names)
print(Bn)
Bn.fix_all()
Bn.unfix("vns(0,1)")

desired_volume = 3.32
volume_weight = 1
term1 = (equil.volume, desired_volume, volume_weight)

desired_iota = -4.26895384431299e-01
iota_weight = 1
term2 = (equil.iota, desired_iota, iota_weight)

prob = LeastSquaresProblem.from_tuples([term1, term2])

if mpi is None:
    least_squares_serial_solve(prob)
else:
    least_squares_mpi_solve(prob, mpi, grad=True)


print("At the optimum,")
print(" volume, according to SPEC    = ", equil.volume())
print(" iota on axis = ", equil.iota())
print(" objective function = ", prob.objective())

# Run the final iteration with a higher poincare resolution
# equil.lib.inputlist.nptrj = 16
# equil.lib.inputlist.nppts = 1000
# equil.lib.inputlist.odetol = 1e-8
# equil.run()
equil.results.plot_poincare()
equil.results.plot_iota()
equil.results.plot_kam_surface()
