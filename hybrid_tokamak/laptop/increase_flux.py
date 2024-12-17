
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
from monty.tempfile import ScratchDir
mpi = MpiPartition()

util.log(logging.INFO)

"""
SPEC constrains the flux per volume in the input file. When running fixed boundary, 
this is arbitrary up to scaling, but in free-boundary it must be consistent with the 
magnetic field strength on the computational boundary. To increase the minor radius of 
a free-boundary SPEC equilibrium, we cannot simply change the geometry, instead we must 
increase the flux and resolve the force balance again. This script does that.
"""

if len(sys.argv)>2:
    # User called the script with arguments, just plot the results
    filename = sys.argv[1]
    with SpecRename(filename) as specf:
        equil = SpecBackoff(specf, mpi, verbose=True, tolerance=1e-10)
        equil.inputlist.nppts = 512
        equil.inputlist.nptrj[0] = [16]
        equil.inputlist.nptrj[1] = [32] 
        equil.inputlist.odetol = 1e-7
        phiedge = equil.inputlist.phiedge
    target_flux = float(sys.argv[2])
    print(f"trying to increase flux from {phiedge} => {target_flux}")
else:
    raise RuntimeError("Must specify a filename and a flux")

assert equil.freebound 
equil.fix_all()
equil.unfix("phiedge")
with ScratchDir(".fluxincrease") as tmpdir: 
    # prob = objectives.ConstrainedProblem(qs.total, tulples_nlc)
    util.proc0_print(f"Free dofs of problem", equil.dof_names)
    step_size = 1e-3
    steps = int(np.ceil((target_flux - phiedge) / step_size))
    for i in range(1,steps+1):
        target_phi = min(target_flux, phiedge + i * step_size)
        equil.boundary.scale(np.sqrt(target_phi/equil.x[0]))
        print("initial guess", equil.initial_guess)
        equil.initial_guess = [equil.boundary.copy()]
        equil.x = np.array([target_phi])
        equil.run()
        print("Aspect ratio now at", equil.boundary.aspect_ratio())
