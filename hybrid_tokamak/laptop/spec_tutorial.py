from simsopt.mhd import Spec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve
from simsopt import geo

equil = Spec("hybrid_tokamak/laptop/hybrid_tokamak.sp")
assert not equil.lib.inputlist.lfreebound, "SPEC must be in Fixed boundary mode"


surf = geo.SurfaceRZFourier.from_vmec_input("hybrid_tokamak/input.hybrid_tokamak")
msurf = geo.SurfaceRZFourier.from_vmec_input("hybrid_tokamak/input.hybrid_tokamak")
msurf.change_resolution(msurf.mpol, 0)
msurf.scale(1.5)

equil.boundary = surf
surf = equil.boundary
surf.fix_all()
surf.unfix("rc(1,1)")
# surf.unfix("zs(1,1)")


desired_volume = 3.32
volume_weight = 1
term1 = (equil.volume, desired_volume, volume_weight)

desired_iota = 4.26895384431299e-01
iota_weight = 1
term2 = (equil.iota, desired_iota, iota_weight)

prob = LeastSquaresProblem.from_tuples([term1, term2])

least_squares_serial_solve(prob)

equil.lib.inputlist.nptrj = 16
equil.lib.inputlist.nppts = 1000
equil.lib.inputlist.odetol = 1e-8
equil.run()

print("At the optimum,")
print(" rc(m=1,n=1) = ", surf.get_rc(1, 1))
print(" zs(m=1,n=1) = ", surf.get_zs(1, 1))
print(" volume, according to SPEC    = ", equil.volume())
print(" volume, according to Surface = ", surf.volume())
print(" iota on axis = ", equil.iota())
print(" objective function = ", prob.objective())

import matplotlib.pyplot as plt

equil.results.plot_poincare()
equil.results.plot_iota()
equil.results.plot_kam_surface()
# equil.results.plot_pressure()
# plt.show()
geo.plot([surf, msurf], engine="plotly")
plt.show()
