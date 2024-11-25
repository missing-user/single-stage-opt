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


surfs = np.linspace(0.001, 1, 64)
surfs = np.array([0.25    ,0.5    ,0.75      ,1.000000000000000E+00])

spec = mhd.Spec("../../SPEC/InputFiles/Verification/forcefree/solovev/solovev_fixed.sp", verbose=False)
# spec = mhd.Spec("../../SPEC/InputFiles/Verification/MHD_stability/w7x_sm_099.sp", verbose=False)

qs_spec = mhd.QuasisymmetryRatioResidualSpec(spec, surfs)
vmec = mhd.Vmec(
    "../../SPEC/InputFiles/Verification/forcefree/solovev/input.solovev_vmec", verbose=False
)
# spec.boundary = vmec.boundary

qs_vmec = mhd.QuasisymmetryRatioResidual(vmec, surfs)
# vmec.boundary.plot(show=False)
# spec.boundary.plot()
# plt.show()
print(surfs)
print(spec.inputlist.nvol)

intermediates = qs_spec.compute()
def dVds(equil:mhd.Spec|mhd.Vmec, intermediates):
    from scipy import integrate
    
    plt.figure()
    dVds = equil.boundary.nfp * integrate.simpson( y=integrate.simpson( y=intermediates.sqrtg, x=intermediates.phi1d ), x=intermediates.theta1d ) 
    if isinstance(equil, mhd.Spec):
        dVds *= intermediates.ds
        vols = np.array([equil.results.get_volume(ivol, ns=32) for ivol in range(equil.nvol)]) 
        print("Volume Using py_spec",sum(vols), vols)
        flux = np.atleast_1d(equil.inputlist.tflux[: spec.nvol])
        # In Fortran tflux is normalized so that tflux(Nvol) = 1.0, so tflux[-1]= 
        plt.scatter(flux / flux[-1], np.cumsum(vols))
    print("spec volume from sqrtg", integrate.simpson(y=dVds, x=surfs ))

    plt.plot(surfs, dVds)
    plt.xlabel("s")
    plt.ylabel("dVds")
    plt.title("dVds"+str(equil))

dVds(spec, intermediates)
print("spec qs", intermediates.total)
plt.figure()
intermediates_vmec = qs_vmec.compute()
dVds(vmec, intermediates_vmec)
print("vmec qs", intermediates_vmec.total)
# spec.results.plot_kam_surface()
spec.results.plot_poincare()
# vmec.boundary.plot()

print(spec.vacuum_well())
print(vmec.vacuum_well())
plt.show()

# plt.subplot(121)
p1 = qs_vmec.profile()
plt.figure()
plt.plot(p1)
plt.title("vmec")
# plt.subplot(122)
p2 = qs_spec.profile()
plt.figure()
plt.plot(p2)
plt.title("spec")
plt.show()

