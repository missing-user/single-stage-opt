import simsopt
from simsopt import mhd
from simsopt import geo
from simsopt.objectives import LeastSquaresProblem, ConstrainedProblem
import simsopt.objectives
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
import matplotlib.pyplot as plt
import numpy as np
from simsopt.util import MpiPartition

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

rotating_ellipse = True
mpi = MpiPartition()
if rotating_ellipse:
    # equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb.sp", mpi, verbose=True)
    equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb_low.sp", mpi, verbose=True, tolerance=1e-11, keep_all_files=False) 
else:
    equil = mhd.Spec("hybrid_tokamak/laptop/nfp2_QA_iota0.4_Vns.sp", mpi, verbose=False, tolerance=1e-11)
assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"
surf = equil.boundary

vmec = mhd.Vmec("hybrid_tokamak/laptop/input.rot_ellipse", mpi, verbose=False)
vmec.boundary = surf
qs = mhd.QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1, 1, 16), helicity_m=1, helicity_n=-1, ntheta=32, nphi=32)
inputlist = equil.lib.inputlist
Bn = equil._normal_field  # This is our new fancy-pants degree of freedom :)

# print("All possible B.n dofs", equil.dof_names)
equil.boundary.fix_all()
Bn.fix_all()
R0 = equil.boundary.major_radius()
for mmax in range(1, 6):
    nmax = mmax
    # nmax = 0
    Bn.fixed_range(0, mmax, -nmax, nmax, False)
    Bn.upper_bounds = np.ones(Bn.local_dof_size) *  2e-2/mmax # higher fourier modes crash the simulation more easily
    Bn.lower_bounds = np.ones(Bn.local_dof_size) * -2e-2/mmax # higher fourier modes crash the simulation more easily
    print("Bn.bounds", Bn.bounds)
    print(qs.profile())
    initial_volume = equil.boundary.volume()
    prob = LeastSquaresProblem.from_tuples(
        [
            (equil.volume, initial_volume, 1),
            # (equil.boundary.major_radius, R0, 1), # try to keep the major radius fixed
            (vmec.vacuum_well, -0.05, 1),
            (qs.residuals, 0, 1),
        ]
    )

    # prob = ConstrainedProblem(
    #     qs.total, 
    #     tuple_lc=,
    #     tuples_nlc=
    # )
    
    print(f"Free dofs of problem", prob.dof_names)

    least_squares_mpi_solve(prob, mpi, abs_step=1e-6, grad=True, save_residuals=True, max_nfev=25)

    equil.results.plot_kam_surface()
    plt.show()

print("At the optimum,")
print(" iota on axis       =", equil.iota())
print(" objective function =", prob.objective())
print(" equil.volume       =", equil.volume())
print(" boundary.R0        =", equil.boundary.major_radius()) 
print(" vmec.vacuum_well   =", vmec.vacuum_well())
print(" qs.profile         =", qs.profile())

prob.plot_graph(show=False)

# Run the final iteration with a higher poincare resolution
equil.lib.inputlist.nptrj = 16
equil.lib.inputlist.nppts = 256
equil.lib.inputlist.odetol = 1e-8
equil.run()

if mpi.proc0_world:
    equil.results.plot_poincare()
    equil.results.plot_iota()
    equil.results.plot_kam_surface()
    equil.results.plot_modB(np.linspace(-0.999, 1, 32), np.linspace(0, 2*np.pi, 32), )
    plt.figure()
    plt.plot(qs.profile())
    plt.title("Quasisymmetry Profile")
    plt.xlabel("s")
    plt.figure()
    j_dot_B, _, _ = equil.results.get_surface_current_density(1)
    plt.subplot(1, 2, 1)
    plt.imshow(j_dot_B[0, 0], origin="lower")
    plt.subplot(1, 2, 2)
    plt.imshow(j_dot_B[0, 1], origin="lower")
    plt.title("Surface current density")

    equil.boundary.plot()

    plt.show()
