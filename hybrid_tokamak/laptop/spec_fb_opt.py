
import glob
import os
import simsopt
import datetime
from simsopt import mhd
from simsopt import geo
from simsopt import objectives
from simsopt.solve import least_squares_mpi_solve, constrained_mpi_solve
import matplotlib.pyplot as plt
import numpy as np
from simsopt.util import MpiPartition
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry
import subprocess
import sys
import logging
from simsopt import util
from hybrid_tokamak.laptop.spec_rename import SpecRename
from hybrid_tokamak.laptop.spec_backoff import SpecBackoff
import py_spec.output
mpi = MpiPartition()

util.log(logging.INFO)

class VmecSpecDependency(mhd.Vmec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # Simsopt SurfaceRZFourier apparently wasn't intended to be used as an output of an Optimizable, so the dependency isn't propagated automatically.
        # I suspect this is a bug, but for now we'll just manually propagate the dependency
        for parent in self.boundary.parents:
            parent.run()
            logging.debug(f"Parent {parent} has been forced to run by {self}")
        
        super().run(*args, **kwargs)

only_plot = False
freeboundary = False
if len(sys.argv)>=2:
    # User called the script with arguments, just plot the results
    filename = sys.argv[1]
    if filename.endswith(".h5"):
        results = py_spec.output.SPECout(filename)
        phiedge = results.input.physics.phiedge
        surf = mhd.Spec.pyspec_to_simsopt_surf(results, 0)
        only_plot = True
    elif filename == "--freeboundary":
        freeboundary = True
        print("FREEBOUNDARY")
    else:
        with SpecRename(filename) as specf:
            equil = SpecBackoff(specf, mpi, verbose=True, tolerance=1e-10)
            phiedge = equil.inputlist.phiedge
            equil.run()
            surf = equil.boundary
            results = equil.results
            only_plot = True
    

if not only_plot:
    equil = SpecBackoff("hybrid_tokamak/laptop/rotating_ellipse_fb_low.sp", mpi, verbose=False, tolerance=1e-10, keep_all_files=True)

    equil.lib.inputlist.odetol = 1e-6
    equil.lib.inputlist.nptrj[0] = 8
    equil.lib.inputlist.nptrj[1] = 2
    equil.lib.inputlist.nppts = 32
    phiedge = equil.inputlist.phiedge

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

vmec = VmecSpecDependency("hybrid_tokamak/laptop/input.rot_ellipse", mpi, verbose=False)
vmec.boundary = surf
vmec.indata.phiedge = phiedge
qs = mhd.QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1, 1, 16), helicity_m=1, helicity_n=0, ntheta=32, nphi=32)
boozer = mhd.Boozer(vmec)
qs2 = mhd.Quasisymmetry(boozer, 
                   0.5, # Radius to target
                   helicity_m=1, helicity_n=0) # (M, N) you want in |B|

# Generate timestamp string for folder name 
timestampdir = datetime.datetime.now().strftime(("freeb_" if freeboundary else "fixb_") +"%m-%d-%H-%M-%S")
subprocess.check_output(["mkdir","-p", timestampdir])

if not only_plot:
    surf.fix_all()
    R0 = 1 #surf.major_radius()
    # equil.unfix("phiedge")

    if freeboundary:
        Bn = equil.normal_field  # This is our new fancy-pants degree of freedom :)
        Bn.fix_all()

        for mmax in range(1, 3):
            nmax = mmax
            # set appropriate bounds for DOFs
            prev_dofs = np.array(Bn.local_dofs_free_status, dtype=bool).copy()
                
            # Bn.fixed_range(0, mmax, -nmax, nmax, False) # unfix square region
            additional_dofs = np.logical_and(Bn.local_dofs_free_status, np.logical_not(prev_dofs))
            # higher fourier modes crash the simulation more easily
            Bn.local_full_lower_bounds = np.where(additional_dofs, Bn.local_full_x - np.ones_like(Bn.local_full_x) * 5e-2/nmax, Bn.local_full_lower_bounds)
            Bn.local_full_upper_bounds = np.where(additional_dofs, Bn.local_full_x + np.ones_like(Bn.local_full_x) * 5e-2/nmax, Bn.local_full_upper_bounds)

    for mmax in range(3, 6):
        nmax = mmax
        if freeboundary:
            # set appropriate bounds for DOFs
            prev_dofs = np.array(Bn.local_dofs_free_status, dtype=bool).copy()
            Bn.fixed_range(0, mmax, -nmax, nmax, False) # unfix square region
            # Bn.fixed_range(0, (mmax-1), -(nmax-1), (nmax-1), True) # fix previous degrees
            additional_dofs = np.logical_and(Bn.local_dofs_free_status, np.logical_not(prev_dofs))
            # higher fourier modes crash the simulation more easily
            Bn.local_full_lower_bounds = np.where(additional_dofs, Bn.local_full_x - np.ones_like(Bn.local_full_x) * 4e-2, Bn.local_full_lower_bounds)
            Bn.local_full_upper_bounds = np.where(additional_dofs, Bn.local_full_x + np.ones_like(Bn.local_full_x) * 4e-2, Bn.local_full_upper_bounds)
            logging.info(f"upper: {Bn.upper_bounds}")
            logging.info(f"value: {Bn.x}")
            logging.info(f"lower: {Bn.lower_bounds}")
        else:
            surf.fixed_range(0, mmax, -nmax, nmax, False)
            surf.lower_bounds = -0.3 * np.ones_like(surf.lower_bounds)
            surf.upper_bounds =  0.3 * np.ones_like(surf.upper_bounds)
            surf.fix("rc(0,0)")
        
        objs = [
                (vmec.mean_iota, 0.4384346834911653, 1),  
                (qs.residuals, 0, 10),
                # (qs2.residuals, 0, 10)
            ]
        tulples_nlc = [
            (vmec.vacuum_well, -1, -0.005)
        ]
        if freeboundary:
            # try to keep the major radius fixed
            R0func = simsopt.make_optimizable(lambda surf: surf.get_rc(0,0), surf)
            objs.append((R0func.J, R0, 3))
        else:
            # Since flux isn't constrained, we must fix the aspect ratio
            objs.append((vmec.aspect, 30, 1))
            tulples_nlc.append((vmec.aspect, 29, 31))
            tulples_nlc.append((surf.major_radius, R0-1e-2, R0+1e-2))

        prob = objectives.LeastSquaresProblem.from_tuples(objs)
        # prob = objectives.ConstrainedProblem(qs.total, tulples_nlc)
        util.proc0_print(f"Free dofs of problem", prob.dof_names)
        kwargs = { }
        if freeboundary:
            kwargs["abs_step"] = 2e-6
            kwargs["xtol"] = 5e-06
            # Larger steps in the magnetic field modes are required to get clean gradients
        if isinstance(prob, objectives.ConstrainedProblem):
            constrained_mpi_solve(prob, mpi, grad=True, options={"maxiter":140}, **kwargs)
        else:
            if freeboundary:
                kwargs["max_nfev"] = 30
            
            least_squares_mpi_solve(prob, mpi, grad=True, **kwargs)
        
        if mpi.proc0_world:
            destpath = os.path.join(timestampdir, f"mpol{mmax}/")
            if freeboundary:
                srcpath ="hybrid_tokamak/laptop/rotating_ellipse_fb_low_000_*"
            else:
                srcpath = "input.rot_ellipse_*"
            subprocess.check_call(["mkdir", "-p", destpath])
            for filename in glob.glob(srcpath):
                subprocess.check_call(["cp", filename, destpath])
        
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
    if freeboundary:
        inputlist.nptrj[1] = 32
    inputlist.nppts = 512
    inputlist.odetol = 1e-7
    equil.recompute_bell()
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
