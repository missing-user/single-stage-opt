import simsopt
from simsopt import mhd
from simsopt import geo
from simsopt.objectives import LeastSquaresProblem
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
    equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb_low.sp", mpi, verbose=True, keep_all_files=True, tolerance=1e-11)
else:
    equil = mhd.Spec("hybrid_tokamak/laptop/nfp2_QA_iota0.4_Vns.sp", mpi, verbose=False, tolerance=1e-11)
assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"
surf = equil.boundary

vmec = mhd.Vmec("hybrid_tokamak/laptop/input.rot_ellipse", mpi, verbose=False)
vmec.boundary = surf
boozer = mhd.Boozer(vmec)
qs = mhd.QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1, 1, 12), helicity_m=1, helicity_n=1, ntheta=32, nphi=32)

inputlist = equil.lib.inputlist
Bn = equil._normal_field  # This is our new fancy-pants degree of freedom :)

# print("All possible B.n dofs", equil.dof_names)
equil.boundary.fix_all()
# equil.fix_all()
Bn.fix_all()
for mmax in range(1, 6):
    nmax = mmax
    # nmax = 0
    Bn.fixed_range(0, mmax, -nmax, nmax, False)
    for key in Bn.dof_names:
        Bn.set_upper_bound(key,  1e-1)
        Bn.set_lower_bound(key, -1e-1)
    
    initial_volume = equil.boundary.volume()
    def callback(equil, vmec, qs):
        print(equil.volume(),"vmec.vacuum_well", vmec.vacuum_well(),"qs.profile", qs.profile())
        return 0
    prob = LeastSquaresProblem.from_tuples(
        [
            (equil.volume, 1.2*initial_volume, 2),
            (vmec.vacuum_well, -0.05, 1),
            (qs.profile, 0, 2),
            (simsopt.make_optimizable(callback, equil, vmec, qs).J , 0, 1)
        ]
    )
    
    print(f"Free dofs of problem", prob.dof_names)

    if mpi is None:
        least_squares_serial_solve(prob)
    else:
        least_squares_mpi_solve(prob, mpi, grad=True, save_residuals=True)

print("At the optimum,")
print(" volume, according to SPEC    = ", equil.volume())
print(" iota on axis = ", equil.iota())
print(" objective function = ", prob.objective())

prob.plot_graph(show=False)
plt.figure()

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
    equil.results.plot_modB(np.linspace(-0.999, 1, 32), np.linspace(0, 2*np.pi, 32), )

    plt.figure()
    j_dot_B, _, _ = equil.results.get_surface_current_density(1)
    plt.subplot(1, 2, 1)
    plt.imshow(j_dot_B[0, 0], origin="lower")
    plt.subplot(1, 2, 2)
    plt.imshow(j_dot_B[0, 1], origin="lower")
    plt.title("Surface current density")

    equil.boundary.plot()

    plt.show()
