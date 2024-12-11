import numpy as np
import py_spec
import sys
from spec_rename import SpecRename
import matplotlib.pyplot as plt

filenames = sys.argv[1:]
fig, axs = plt.subplots(2, 3, figsize=(12, 4))
axs = axs.flatten()

colors = plt.cm.tab20b(np.linspace(0, 1, len(filenames)))
colors = colors[:len(filenames)]
for filename, c in zip(filenames, colors):
  try:
    out = py_spec.SPECout(filename) 
  except:
    print("Failed to read ", filename)
    continue
  
  for i, ax in enumerate(axs):  
    out.plot_kam_surface(ntheta=128, ax=ax, c=c[np.newaxis, :], label=filename, zeta=i*np.pi/len(axs))
    ax.set_xlabel("s")
    ax.set_ylabel("z")
    ax.set_title(f"Slice $\zeta = {i/len(axs):.2f} \pi$")
    if i == 0:
      ax.legend(prop={'size': 6})
# plt.legend(filenames)
plt.show()

for filename in filenames:
  try:
    out = py_spec.SPECout(filename) 
  except:
    print("Failed to read ", filename)
    continue

  fig, axs = plt.subplots(2, 3, sharex=True,  figsize=(12, 4))
  axs = axs.flatten()
  out.plot_kam_surface(ax=axs[0])
  out.plot_iota(ax=axs[1])
  out.plot_modB(np.linspace(-1+1e-6, 1, 64), np.linspace(0, 2*np.pi, 64), lvol=[0], ax=axs[2])
  def angle2idx(angle):
    return int(np.floor(angle / (2*np.pi) * out.poincare.R.shape[2]))
  out.plot_poincare(ax=axs[3], s=1, )
  out.plot_poincare(ax=axs[4], s=1, toroidalIdx=angle2idx(0.5*np.pi))
  out.plot_poincare(ax=axs[5], s=1, toroidalIdx=angle2idx(np.pi))
  
plt.show()
