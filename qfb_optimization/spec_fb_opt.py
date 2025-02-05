
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
from qfb_optimization.spec_rename import SpecRename
from qfb_optimization.spec_backoff import SpecBackoff
import py_spec.output

import qfb_optimization.latexplot as latexplot
mpi = MpiPartition()

util.log(logging.INFO)

"""
Run either fixed or quasi-free-boundary spec optimization to compare the results:
mpiexec -n 4 -map-by node:PE=1 --display map bash -c 'python -m qfb_optimization.spec_fb_opt'
or:
mpiexec -n 4 -map-by node:PE=1 --display map bash -c 'python -m qfb_optimization.spec_fb_opt --freeboundary'
"""

class VmecSpecDependency(mhd.Vmec):
    def run(self, *args, **kwargs):
        # Simsopt SurfaceRZFourier apparently wasn't intended to be used as an output of an Optimizable, so the dependency isn't propagated automatically.
        # I suspect this is a bug, but for now we'll just manually propagate the dependency
        for parent in self.boundary.parents:
            parent.run()
            logging.debug(f"Parent {parent} has been forced to run by {self}")
        
        return super().run(*args, **kwargs)

only_plot = False
freeboundary = False
if len(sys.argv)>=2:
    # User called the script with arguments, just plot the results
    filename = sys.argv[1]
    if filename == "--freeboundary":
        freeboundary = True
        print("FREEBOUNDARY")
    else:
        with SpecRename(filename) as specf:
            equil = SpecBackoff(specf, mpi, verbose=True, tolerance=1e-10)
            phiedge = equil.inputlist.phiedge
            if filename.endswith(".h5"):
                results = py_spec.output.SPECout(filename)
                phiedge = results.input.physics.phiedge
                surf = mhd.Spec.pyspec_to_simsopt_surf(results, 0)
                aspect_target = surf.aspect_ratio()
                only_plot = True
            else:
                equil.run()
                surf = equil.boundary
                results = equil.results
                aspect_target = surf.aspect_ratio()
                only_plot = True
    

if not only_plot:
    equil = SpecBackoff("qfb_optimization/rotating_ellipse_fb_low.sp", mpi, verbose=True, tolerance=1e-10, keep_all_files=False, max_attempts=3)

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
    aspect_target = surf.aspect_ratio()

vmec = VmecSpecDependency("qfb_optimization/input.rot_ellipse", mpi, verbose=False, keep_all_files=False)
vmec.boundary = surf
vmec.indata.phiedge = phiedge
qs = mhd.QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1, 1, 16), helicity_m=1, helicity_n=0, ntheta=32, nphi=32)
boozer = mhd.Boozer(vmec)
qs2 = mhd.Quasisymmetry(boozer, 
                   0.5, # Radius to target
                   helicity_m=qs.helicity_m, helicity_n=qs.helicity_n) # (M, N) you want in |B|

# Generate timestamp string for folder name 
timestampdir = datetime.datetime.now().strftime(("freeb_" if freeboundary else "fixb_") +"%m-%d-%H-%M-%S")


def make_objs():
    objs = [
            (vmec.mean_iota, 0.4384346834911653, 0.1),  
            (qs.residuals, 0, 1),
            # (qs2.residuals, 0, 10)
        ]
    if freeboundary:
        # try to keep the major radius fixed
        R0func = simsopt.make_optimizable(lambda surf: surf.get_rc(0,0), surf)
        objs.append((R0func.J, R0, 1))
    else:
        # Since flux isn't constrained, we must fix the aspect ratio
        objs.append((vmec.aspect, aspect_target, 1))
    return objs

if not only_plot:
    subprocess.check_output(["mkdir","-p", timestampdir])
    surf.fix_all()
    R0 = 1 #surf.major_radius()
    # equil.unfix("phiedge")

    if freeboundary:
        Bn = equil.normal_field  # This is our new fancy-pants degree of freedom :)
        Bn.fix_all()

        # for mmax in range(1, 3):
        #     nmax = mmax
        #     # set appropriate bounds for DOFs
        #     prev_dofs = np.array(Bn.local_dofs_free_status, dtype=bool).copy()
                
        #     # Bn.fixed_range(0, mmax, -nmax, nmax, False) # unfix square region
        #     additional_dofs = np.logical_and(Bn.local_dofs_free_status, np.logical_not(prev_dofs))
        #     # higher fourier modes crash the simulation more easily
        #     Bn.local_full_lower_bounds = np.where(additional_dofs, Bn.local_full_x - np.ones_like(Bn.local_full_x) * 5e-2/nmax, Bn.local_full_lower_bounds)
        #     Bn.local_full_upper_bounds = np.where(additional_dofs, Bn.local_full_x + np.ones_like(Bn.local_full_x) * 5e-2/nmax, Bn.local_full_upper_bounds)

    for mmax in range(1, 6):
        nmax = mmax
        if freeboundary:
            # set appropriate bounds for DOFs
            prev_dofs = np.array(Bn.local_dofs_free_status, dtype=bool).copy()
            Bn.fixed_range(0, mmax, -nmax, nmax, False) # unfix square region
            # Bn.fixed_range(0, (mmax-1), -(nmax-1), (nmax-1), True) # fix previous degrees
            additional_dofs = np.logical_and(Bn.local_dofs_free_status, np.logical_not(prev_dofs))
            # higher fourier modes crash the simulation more easily
            Bn.local_full_lower_bounds = np.where(additional_dofs, Bn.local_full_x - np.ones_like(Bn.local_full_x) * 20e-2, Bn.local_full_lower_bounds)
            Bn.local_full_upper_bounds = np.where(additional_dofs, Bn.local_full_x + np.ones_like(Bn.local_full_x) * 20e-2, Bn.local_full_upper_bounds)
            logging.info(f"upper: {Bn.upper_bounds}")
            logging.info(f"value: {Bn.x}")
            logging.info(f"lower: {Bn.lower_bounds}")
        else:
            surf.fixed_range(0, mmax, -nmax, nmax, False)
            surf.lower_bounds = -0.3 * np.ones_like(surf.lower_bounds)
            surf.upper_bounds =  0.3 * np.ones_like(surf.upper_bounds)
            surf.fix("rc(0,0)")
        
        objs = make_objs() 
        prob = objectives.LeastSquaresProblem.from_tuples(objs)
        # latexplot.figure()
        # prob.plot_graph(show=False)
        # latexplot.savenshow("dependency_graph")

        
        # prob = objectives.ConstrainedProblem(qs.total, tulples_nlc)
        util.proc0_print(f"Free dofs of problem", prob.dof_names)
        kwargs = { }
        if freeboundary:
            # Larger steps in the magnetic field modes are required to get clean gradients
            kwargs["abs_step"] = 8e-7
            kwargs["xtol"] = 1e-06
            kwargs["ftol"] = 1e-4
        
        if freeboundary:
            kwargs["max_nfev"] = 30

        if not freeboundary:
            kwargs["ftol"] = 1e-4
        
        least_squares_mpi_solve(prob, mpi, grad=True, **kwargs)
        
        if mpi.proc0_world:
            destpath = os.path.join(timestampdir, f"mpol{mmax}/")
            if freeboundary:
                srcpath ="qfb_optimization/rotating_ellipse_fb_low_000_*"
            else:
                srcpath = "input.rot_ellipse_*"
            subprocess.check_call(["mkdir", "-p", destpath])
            for filename in glob.glob(srcpath):
                subprocess.check_call(["mv", filename, destpath])
        
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
util.proc0_print(" qs.total         =", qs.total())
util.proc0_print(" qs2.profile        =", qs2.J())
util.proc0_print(" qs2.norm        =", np.linalg.norm(qs2.J()))
util.proc0_print(" LgradB             =", getLgradB(vmec))

if only_plot:
    objs = make_objs()
    prob = objectives.LeastSquaresProblem.from_tuples(objs)
    util.proc0_print(" objective function =", prob.objective())


# Run the final iteration with a higher poincare resolution
if not only_plot:
    inputlist = equil.lib.inputlist
    if freeboundary:
        inputlist.nptrj[0] = 16
        inputlist.nptrj[1] = 32
    inputlist.nppts = 512
    inputlist.odetol = 1e-7
    equil.recompute_bell()
    equil.run()

if mpi.proc0_world:
    if not only_plot:
        results = equil.results
    geo.plot([surf, equil.computational_boundary.copy(range="field period")], close=True, engine="plotly")
    # surf.plot(engine="plotly")
    # latexplot.savenshow(filename+"_kam")
    latexplot.set_cmap(8)
    results.plot_kam_surface(linewidth=1, c="black")
    results.plot_poincare(ax=plt.gca())
    plt.legend(["magnetic axis", "$\mathcal{D}$", "$\mathcal{S}$",  "Poincare trace"])
    latexplot.savenshow(filename+"_poincare")
    results.plot_iota()
    latexplot.savenshow(filename+"_iotaprofile")
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
    latexplot.savenshow(filename+"_qsprofile")

    # Copy the poincare stuff
    destpath = os.path.join(timestampdir, "")
    if freeboundary:
        srcpath ="qfb_optimization/rotating_ellipse_fb_low_000_*"
    else:
        srcpath = "input.rot_ellipse_*"
    subprocess.check_call(["mkdir", "-p", destpath])
    for filename in glob.glob(srcpath):
        subprocess.check_call(["mv", filename, destpath])