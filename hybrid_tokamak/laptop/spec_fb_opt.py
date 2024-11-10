from simsopt.mhd import Spec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
from simsopt import geo
from simsopt.util import MpiPartition
import matplotlib.pyplot as plt
import numpy as np
import warnings

mpi = MpiPartition(ngroups=4)


equil = Spec("hybrid_tokamak/laptop/hybrid_tokamak_vacuum.sp")
assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"
if equil.lib.inputlist.linitialize != 0:
    warnings.warn(
        "I would recommend setting linitialize to 1 or a negative value, "
        "instead of manually manually initializing the surfaces."
    )

Bn = equil.normal_field  # This is our new fancy-pants degree of freedom :)

msurf = geo.SurfaceRZFourier.from_vmec_input("hybrid_tokamak/input.hybrid_tokamak")
msurf.change_resolution(msurf.mpol, 0)
msurf.scale(1.5)
msurf.change_resolution(equil.lib.inputlist.mpol, equil.lib.inputlist.ntor)

equil.normal_field.surface = msurf

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
