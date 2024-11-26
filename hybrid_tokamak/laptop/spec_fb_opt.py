from simsopt import mhd
from simsopt import geo
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
import matplotlib.pyplot as plt
import numpy as np
from simsopt.util import MpiPartition
import hybrid_tokamak.generate_Bn_initial as Bn_initial

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

mpi = MpiPartition(2)
equil = mhd.Spec.default_freeboundary(copy_to_pwd=True)
rotating_ellipse = True
if rotating_ellipse:
    # equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb.sp", mpi, verbose=True)
    equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb.sp", mpi, verbose=True)
else:
    equil = mhd.Spec("hybrid_tokamak/laptop/nfp2_QA_iota0.4_Vns.sp", mpi, verbose=True)
assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"

vmec = mhd.Vmec("hybrid_tokamak/laptop/input.rot_ellipse", mpi)
vmec.boundary = equil.boundary
boozer = mhd.Boozer(vmec)
qs = mhd.QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1, 1, 12), helicity_m=1, helicity_n=1, ntheta=32, nphi=32)

inputlist = equil.lib.inputlist
inputlist.lautoinitbn = False
inputlist.mfreeits = 8

# To allow for poincare tracing
inputlist.nppts = 32
for ivol in range(equil.mvol):
    inputlist.nptrj[ivol] = 8

Bn = equil.normal_field  # This is our new fancy-pants degree of freedom :)

print(equil.dof_names)
print("B.n", Bn)
equil.fix_all()
Bn.fix_all()
for mmax in range(1, equil.mvol):
    nmax = mmax
    Bn.fixed_range(0, mmax, -nmax, nmax, False)

    if False:
        # Run the final iteration with a higher poincare resolution
        # equil.tflux_profile.fix_all()
        # equil.iota_profile.fix_all()
        # equil.oita_profile.fix_all()


        Bn.surface.plot(show=False, close=True)
        equil.boundary.plot(show=False, close=True)

        equil.results.plot_modB(np.linspace(-0.999, 1, 32), np.linspace(0, 2*np.pi, 32), )
        equil.results.plot_poincare()
        equil.results.plot_kam_surface()
        plt.show()


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
    initial_volume = equil.boundary.volume()
    prob = LeastSquaresProblem.from_tuples(
        [
            (qs.residuals, 0, 1),
            (equil.volume, initial_volume, 1)
        ]
    )

    if mpi is None:
        least_squares_serial_solve(prob)
    else:
        least_squares_mpi_solve(prob, mpi, grad=True)

    equil.save(f"hybrid_tokamak/laptop/solution{mmax}")

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
