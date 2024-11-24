import simsopt
from simsopt import objectives
from simsopt import mhd
from simsopt import configs  # configs.zoo.get()

from simsopt import objectives
import simsopt.objectives
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
from simsopt import geo
from simsopt.util import MpiPartition
import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

surfs = np.linspace(0.001, 1, 65)
surfs = np.array([0.25    ,0.5    ,0.75      ,1.000000000000000E+00])

spec = mhd.Spec("../../SPEC/InputFiles/Verification/forcefree/solovev/solovev_fixed.sp")

qs_spec = mhd.QuasisymmetryRatioResidualSpec(spec, surfs)
vmec = mhd.Vmec(
    "../../SPEC/InputFiles/Verification/forcefree/solovev/input.solovev_vmec"
)
qs_vmec = mhd.QuasisymmetryRatioResidual(vmec, surfs)
# vmec.boundary.plot(show=False)
# spec.boundary.plot()
# plt.show()
print(surfs)
print(spec.inputlist.nvol)
print("spec", qs_spec.total())
plt.figure()
print("vmec", qs_vmec.total())
plt.figure()
spec.results.plot_kam_surface()
vmec.boundary.plot()

print(spec.vacuum_well())
print(vmec.vacuum_well())
plt.show()

plt.subplot(121)
p1 = qs_vmec.profile()
plt.figure()
plt.plot(p1)
plt.title("vmec")
plt.subplot(122)
p2 = qs_spec.profile()
plt.figure()
plt.plot(p2)
plt.title("spec")
plt.show()

