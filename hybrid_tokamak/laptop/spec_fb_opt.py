from simsopt import mhd
from simsopt import geo
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
import matplotlib.pyplot as plt
import numpy as np
from simsopt.util import MpiPartition

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# equil = mhd.Spec.default_freeboundary(copy_to_pwd=True)
equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb.sp", verbose=True)
mpi = equil.mpi
assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"

inputlist = equil.lib.inputlist
# inputlist.lautoinitbn = True
# inputlist.mfreeits = 1

# To allow for poincare tracing
inputlist.nppts = 64
nvol = equil.nvol
for ivol in range(nvol):
    inputlist.nptrj[ivol] = 8

# inputlist.mpol = 7
# inputlist.ntor = 7

# Solve for iota and oita
inputlist.lconstraint = 1

# def change_nvol(new_nvol):
#     inputlist.nvol = new_nvol
#     equil.nvol = new_nvol
#     equil.mvol = new_nvol + int(equil.freebound)
#     inputlist.linitialize = 1
#     equil.initial_guess = None
# change_nvol(2)

Bn = equil.normal_field  # This is our new fancy-pants degree of freedom :)
for lvol in range(nvol):
    inputlist.lrad[lvol] = 6
Bn.surface.set_rc(1,1,0)
Bn.surface.set_zs(0,1,0)
Bn.surface.set_zs(1,1,0)
Bn.surface.set_zs(0,1,0)
if mpi.proc0_world:
    Bn.surface.plot()
    equil.boundary.plot()
    plt.imshow(Bn.get_vns_asarray())
    plt.show()

equil.run()
# Bn.surface.change_resolution(1,1)


desired_iota = .426895384431299
iota_weight = 1
if mpi.proc0_world:
    # Run the final iteration with a higher poincare resolution
    # equil.tflux_profile.fix_all()
    # equil.iota_profile.fix_all()
    # equil.oita_profile.fix_all()


    Bn.surface.plot()
    equil.boundary.plot()
    plt.imshow(Bn.get_vns_asarray())
    plt.show()

    print(equil.dof_names)
    print("B.n", Bn)
    Bn.fix_all()
    mmax = nmax = 1
    Bn.fixed_range(0, mmax, -nmax, nmax, False)
    equil.results.plot_poincare()
    equil.results.plot_kam_surface()
    


    # iota = p / q
    p = 8
    q = 5
    residue1 = mhd.Residue(equil, p, q, s_guess=0.65)
    print("iota", equil.iota())
    for p,q in [(8,5), (-8,5), (10,5), (7,5), (-7,5), (9,5), (-9,5)]:
        try:
            print("Compute residue for p = ", p, ", q = ", q)
            print(mhd.Residue(equil, p, q, s_guess=0.65).J())
        except:
            print("Failed to compute residue for p = ", p, ", q = ", q)
    exit()

prob = LeastSquaresProblem.from_tuples(
    [
        (equil.iota, desired_iota, iota_weight),
    ]
)

if mpi is None:
    least_squares_serial_solve(prob)
else:
    least_squares_mpi_solve(prob, mpi, grad=True)


print("At the optimum,")
print(" volume, according to SPEC    = ", equil.volume())
print(" iota on axis = ", equil.iota())
print(" objective function = ", prob.objective())

# Run the final iteration with a higher poincare resolution
equil.lib.inputlist.nptrj = 16
equil.lib.inputlist.nppts = 128
equil.lib.inputlist.odetol = 1e-8
equil.run()

if mpi.proc0_world:
    equil.results.plot_poincare()
    equil.results.plot_iota()
    equil.results.plot_pressure()
    equil.results.plot_kam_surface()
    equil.results.plot_modB()

    plt.figure()
    j_dot_B, _, _ = equil.results.get_surface_current_density(1)
    plt.subplot(1, 2, 1)
    plt.imshow(j_dot_B[0, 0], origin="lower")
    plt.subplot(1, 2, 2)
    plt.imshow(j_dot_B[0, 1], origin="lower")
    plt.title("Surface current density")

    equil.boundary.plot()

    plt.show()
