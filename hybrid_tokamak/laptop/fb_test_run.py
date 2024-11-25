from simsopt import mhd
import matplotlib.pyplot as plt
import numpy as np
from simsopt.util import MpiPartition



mpi = MpiPartition(ngroups=4)
equil = mhd.Spec.default_freeboundary(copy_to_pwd=True)
# assert equil.lib.inputlist.lfreebound, "SPEC must be in Freeboundary mode"

input_conf = equil.lib.inputlist
input_conf.lautoinitbn = True
input_conf.mfreeits = 100

# To allow for poincare tracing
input_conf.nppts = 64
input_conf.nptrj = 16

input_conf.mpol = 4
input_conf.ntor = 4

# Solve for iota and oita
input_conf.lconstraint = 1


def change_nvol(new_nvol):
    input_conf.nvol = new_nvol
    equil.nvol = new_nvol
    equil.mvol = new_nvol + int(equil.freebound)
    input_conf.linitialize = 1
    equil.initial_guess = None
change_nvol(2)

nvol = equil.nvol
for lvol in range(nvol):
    input_conf.lrad[lvol] = 6

equil.tflux_profile = mhd.ProfileSpec(np.linspace(1.0/nvol, 1.0, equil.mvol))
equil.tflux_profile.unfix_all()

plt.figure()
s = np.linspace(0, nvol, 100, endpoint=False)
plt.plot(s, equil.tflux_profile.f(s))
plt.title("Toroidal flux profile")

# plt.show()
equil.iota_profile = mhd.ProfileSpec([0.42689]*(equil.mvol+1))
equil.oita_profile = mhd.ProfileSpec([0.42689]*(equil.mvol+1))
equil.iota_profile.unfix_all()
equil.oita_profile.unfix_all()

# Run the final iteration with a higher poincare resolution
equil.run()

# Fallback to reading the results if we skip the run
if not hasattr(equil, "results") or equil.results is None:
    import py_spec

    equil.results = py_spec.SPECout("defaults_freebound_000_000000.sp.h5")
else:
    print("equil.results path:", equil.results.filename)
equil.results.plot_poincare()
equil.results.plot_iota()
equil.results.plot_pressure()
equil.results.plot_kam_surface()
equil.results.plot_modB()


plt.figure()
j_dot_B, _, _ = equil.results.get_surface_current_density(1)
plt.subplot(1, 2, 1)
plt.imshow(j_dot_B[0, 0], origin="lower")
plt.subplot(1, 2, 2)
plt.imshow(j_dot_B[0, 1], origin="lower")
plt.title("Surface current density")

equil.boundary.plot()

plt.show()
