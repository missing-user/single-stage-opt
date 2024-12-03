import simsopt
from simsopt import mhd
from simsopt import util
from simsopt import geo
from simsopt.objectives import LeastSquaresProblem
import simsopt.objectives
from simsopt.solve import least_squares_mpi_solve
import matplotlib.pyplot as plt
import numpy as np
from simsopt.util import MpiPartition
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry

import sys
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

mpi = MpiPartition()
only_plot = False
freeboundary = True
if len(sys.argv)>=2:
    filename = sys.argv[1]
    if filename.endswith(".sp.end"):
        import subprocess
        subprocess.check_call(["cp", filename, filename[:-4]])
        filename = filename[:-4]
    equil = mhd.Spec(filename, mpi, verbose=True)
    only_plot = True
else:
    equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb_low.sp", mpi, verbose=True, tolerance=1e-11, keep_all_files=True)
    

assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"
if freeboundary:
    # equil.activate_profile("tflux")
    equil.activate_profile("pflux")
    surf = equil.boundary
else:
    surf = equil.boundary.copy()

vmec = mhd.Vmec("hybrid_tokamak/laptop/input.rot_ellipse", mpi, verbose=False)
vmec.boundary = surf
qs = mhd.QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1, 1, 16), helicity_m=1, helicity_n=1, ntheta=32, nphi=32)
Bn = equil._normal_field  # This is our new fancy-pants degree of freedom :)
surf.fix_all()
Bn.fix_all()
# equil.tflux_profile.fix("x1")
# equil.tflux_profile.unfix("x0")
# equil.tflux_profile.set_lower_bound("x0", 1)
# equil.pflux_profile.fix("x1")
# equil.pflux_profile.unfix("x0")
# R0 = equil.boundary.major_radius()
for mmax in range(2, 3):
    nmax = mmax
    if freeboundary:
        Bn.fixed_range(0, mmax, -nmax, nmax, False)
        Bn.upper_bounds = np.ones(Bn.local_dof_size) *  2e-2 # higher fourier modes crash the simulation more easily
        Bn.lower_bounds = np.ones(Bn.local_dof_size) * -2e-2 # higher fourier modes crash the simulation more easily
    else:
        surf.fixed_range(0, mmax, -nmax, nmax, False)
    logging.info(f"Aspect ratio is now {vmec.aspect()}")
    initial_volume = surf.volume()
    R0 = surf.major_radius()
    prob = LeastSquaresProblem.from_tuples(
        [
            # (vmec.aspect, 26, 1e-3),
            # (vmec.iota_axis, 0.4384346834911653, 1), 
            # (vmec.iota_edge, 0.4384346834911653, 1),
            # (equil.volume if freeboundary else surf.volume, initial_volume, 1),
            # (surf.major_radius, R0, 1), # try to keep the major radius fixed
            # (vmec.vacuum_well, -0.05, 1),
            # (qs.residuals, 0, 1),

            (equil.volume, initial_volume, 1),
            (equil.boundary.major_radius, R0, 1), # try to keep the major radius fixed
            (vmec.vacuum_well, -0.05, 1),
            (qs.profile, 0, 2),
        ]
    )
    
    util.proc0_print(f"Free dofs of problem", prob.dof_names)
    if not only_plot:
        least_squares_mpi_solve(prob, mpi, abs_step=1e-5, grad=True)


def getLgradB(vmec:mhd.Vmec):
    s = np.linspace(0.25, 1, 16)
    ntheta = 32
    nphi = 32
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
    data = vmec_compute_geometry(vmec, s, theta, phi)
    return np.min(data.L_grad_B)

# Run the final iteration with a higher poincare resolution
inputlist = equil.lib.inputlist
inputlist.nptrj[0] = 16
inputlist.nppts = 256
inputlist.odetol = 1e-8
util.proc0_print("At the optimum,")
util.proc0_print(" objective function =", prob.objective())
util.proc0_print(" iota on axis       =", vmec.iota_axis())
util.proc0_print(" iota edge          =", vmec.iota_edge())
util.proc0_print(" boundary.R0        =", surf.major_radius()) 
util.proc0_print(" aspect ratio       =", surf.aspect_ratio())
util.proc0_print(" equil.volume       =", vmec.volume())
util.proc0_print(" vmec.vacuum_well   =", vmec.vacuum_well())
util.proc0_print(" qs.profile         =", qs.profile())
util.proc0_print(" LgradB             =", getLgradB(vmec))

prob.plot_graph(show=False)
equil.run()
if mpi.proc0_world:
    surf.plot(engine="plotly")
    equil.results.plot_poincare()
    equil.results.plot_iota()
    equil.results.plot_kam_surface()
    # equil.results.plot_modB(np.linspace(-0.999, 1, 32), np.linspace(0, 2*np.pi, 32), )
    plt.figure()
    plt.plot(qs.profile())
    plt.title("Quasisymmetry Profile")
    plt.xlabel("s")
    plt.figure()
    j_dot_B, _, _ = equil.results.get_surface_current_density(1)
    plt.subplot(1, 2, 1)
    plt.imshow(j_dot_B[0, 0], origin="lower")
    plt.title("Surface current density inner")
    plt.subplot(1, 2, 2)
    plt.imshow(j_dot_B[0, 1], origin="lower")
    plt.title("Surface current density outer")

    equil.boundary.plot()
    plt.show()
