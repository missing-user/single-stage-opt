import simsopt
from simsopt import mhd
from simsopt import geo
from simsopt import objectives
from simsopt.solve import least_squares_mpi_solve
import matplotlib.pyplot as plt
import numpy as np
from simsopt.util import MpiPartition
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry
import subprocess
import sys
import logging
from simsopt import util

import py_spec.output
mpi = MpiPartition()
only_plot = False
freeboundary = True
if len(sys.argv)>=2:
    # User called the script with arguments, just plot the results
    filename = sys.argv[1]
    if filename.endswith(".sp.end"):
        import subprocess
        subprocess.check_call(["cp", filename, filename[:-4]])
        filename = filename[:-4]
        equil = mhd.Spec(filename, mpi, verbose=True, tolerance=1e-10)
        phiedge = equil.inputlist.phiedge
        equil.run()
        surf = equil.boundary
        results = equil.results
    elif filename.endswith(".h5"):
        results = py_spec.output.SPECout(filename)
        phiedge = results.input.physics.phiedge
        surf = mhd.Spec.pyspec_to_simsopt_surf(results, 0)
    else:
        raise ValueError("Filename must end with .sp.end or .h5")
    only_plot = True
else:
    equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb_low.sp", mpi, verbose=True, tolerance=1e-10, keep_all_files=True)

    equil.lib.inputlist.odetol = 1e-6
    equil.lib.inputlist.nptrj[0] = 8
    equil.lib.inputlist.nptrj[1] = 2
    equil.lib.inputlist.nppts = 32
    phiedge = equil.inputlist.phiedge
    util.initialize_logging("freeboundary.log", mpi=True, level=logging.INFO)

    if freeboundary:
        # equil.activate_profile("tflux")
        # equil.activate_profile("pflux")
        # equil.tflux_profile.fix("x1")
        # equil.tflux_profile.unfix("x0")
        # equil.tflux_profile.set_lower_bound("x0", 1)
        # equil.pflux_profile.fix("x1")
        # equil.pflux_profile.unfix("x0")
        surf = equil.boundary
        assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"
    else:
        surf = equil.boundary.copy()

vmec = mhd.Vmec("hybrid_tokamak/laptop/input.rot_ellipse", mpi, verbose=False)
vmec.boundary = surf
vmec.indata.phiedge = phiedge
qs = mhd.QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1, 1, 16), helicity_m=1, helicity_n=0, ntheta=32, nphi=32)
boozer = mhd.Boozer(vmec)
qs2 = mhd.Quasisymmetry(boozer, 
                   0.5, # Radius to target
                   helicity_m=1, helicity_n=0) # (M, N) you want in |B|

if not only_plot:
    Bn = equil._normal_field  # This is our new fancy-pants degree of freedom :)
    surf.fix_all()
    Bn.fix_all()
    R0 = 1 #surf.major_radius()
    # equil.unfix("phiedge")

    for mmax in range(1, 3):
        nmax = mmax
        if freeboundary:
            # set appropriate bounds for DOFs
            prev_dofs = np.array(Bn.local_dofs_free_status, dtype=bool).copy()
                
            # Bn.fixed_range(0, mmax, -nmax, nmax, False) # unfix square region
            additional_dofs = np.logical_and(Bn.local_dofs_free_status, np.logical_not(prev_dofs))
            # higher fourier modes crash the simulation more easily
            Bn.local_full_lower_bounds = np.where(additional_dofs, Bn.local_full_x - np.ones_like(Bn.local_full_x) * 3e-2/nmax, Bn.local_full_lower_bounds)
            Bn.local_full_upper_bounds = np.where(additional_dofs, Bn.local_full_x + np.ones_like(Bn.local_full_x) * 3e-2/nmax, Bn.local_full_upper_bounds)

    for mmax in range(3, 5):
        nmax = mmax
        if freeboundary:
            # set appropriate bounds for DOFs
            prev_dofs = np.array(Bn.local_dofs_free_status, dtype=bool).copy()
                
            Bn.fixed_range(0, mmax, -nmax, nmax, False) # unfix square region
            additional_dofs = np.logical_and(Bn.local_dofs_free_status, np.logical_not(prev_dofs))
            # higher fourier modes crash the simulation more easily
            Bn.local_full_lower_bounds = np.where(additional_dofs, Bn.local_full_x - np.ones_like(Bn.local_full_x) * 2e-2, Bn.local_full_lower_bounds)
            Bn.local_full_upper_bounds = np.where(additional_dofs, Bn.local_full_x + np.ones_like(Bn.local_full_x) * 2e-2, Bn.local_full_upper_bounds)
            logging.info(f"upper: {Bn.upper_bounds}")
            logging.info(f"value: {Bn.x}")
            logging.info(f"lower: {Bn.lower_bounds}")
        else:
            surf.fixed_range(0, mmax, -nmax, nmax, False)
            # surf.fix("rc(0,0)")
        
        prob = objectives.LeastSquaresProblem.from_tuples(
            [
                # (equil.volume if freeboundary else surf.volume, initial_volume, 1),
                # (surf.major_radius, R0, 1), # try to keep the major radius fixed
                # (vmec.vacuum_well, -0.05, 1),
                # (qs.residuals, 0, 1),

                (vmec.iota_edge, 0.4384346834911653, 1),
                (vmec.mean_iota, 0.412, 1), 
                (surf.major_radius, R0, 3), # try to keep the major radius fixed
                (vmec.vacuum_well, -0.01, 2),
                (qs.residuals, 0, 1)
            ]
        )
        util.proc0_print(f"Free dofs of problem", prob.dof_names)
        
        least_squares_mpi_solve(prob, mpi, abs_step=1e-6, grad=True, 
                                ftol=1e-06, xtol=5e-06, gtol=1e-06, max_nfev=40)
        
        subprocess.check_call(["cp", "hybrid_tokamak/laptop/rotating_ellipse_fb_low_00*", f"hybrid_tokamak/laptop/working_optimization/boundswradius{mmax}/"])

        util.proc0_print("At the resolution increase")
        util.proc0_print(" objective function =", prob.objective())
        util.proc0_print(" iota on axis       =", vmec.iota_axis())
        util.proc0_print(" iota edge          =", vmec.iota_edge())
        util.proc0_print(" mean_iota          =", vmec.mean_iota())
        util.proc0_print(" boundary.R0        =", surf.major_radius()) 
        util.proc0_print(" aspect ratio       =", surf.aspect_ratio())
        util.proc0_print(" equil.volume       =", vmec.volume())
        util.proc0_print(" vmec.vacuum_well   =", vmec.vacuum_well())
        util.proc0_print(" qs.profile         =", qs.profile())
        util.proc0_print(" qs2.profile        =", qs2.J())  


def getLgradB(vmec:mhd.Vmec):
    s = np.linspace(0.25, 1, 16)
    ntheta = 32
    nphi = 32
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
    data = vmec_compute_geometry(vmec, s, theta, phi)
    return np.min(data.L_grad_B)

util.proc0_print("At the optimum,")
util.proc0_print(" iota on axis       =", vmec.iota_axis())
util.proc0_print(" iota edge          =", vmec.iota_edge())
util.proc0_print(" boundary.R0        =", surf.major_radius()) 
util.proc0_print(" aspect ratio       =", surf.aspect_ratio())
util.proc0_print(" equil.volume       =", vmec.volume())
util.proc0_print(" vmec.vacuum_well   =", vmec.vacuum_well())
util.proc0_print(" qs.profile         =", qs.profile())
util.proc0_print(" qs2.profile        =", qs2.J())
util.proc0_print(" LgradB             =", getLgradB(vmec))

# prob.plot_graph(show=False)

# Run the final iteration with a higher poincare resolution
if not only_plot:
    inputlist = equil.lib.inputlist
    inputlist.nptrj[0] = 16
    inputlist.nppts = 512
    inputlist.odetol = 1e-7
    equil.run()


if mpi.proc0_world:
    if not only_plot:
        results = equil.results
    surf.plot(engine="plotly")
    results.plot_poincare()
    results.plot_iota()
    results.plot_kam_surface()
    # equil.results.plot_modB(np.linspace(-0.999, 1, 32), np.linspace(0, 2*np.pi, 32), )
    plt.figure()
    plt.plot(qs.profile())
    plt.title("Quasisymmetry Profile")
    plt.xlabel("s")
    # plt.figure()
    # j_dot_B, _, _ = results.get_surface_current_density(1)
    # plt.subplot(1, 2, 1)
    # plt.imshow(j_dot_B[0, 0], origin="lower")
    # plt.title("Surface current density inner")
    # plt.subplot(1, 2, 2)
    # plt.imshow(j_dot_B[0, 1], origin="lower")
    # plt.title("Surface current density outer")
    plt.show()
