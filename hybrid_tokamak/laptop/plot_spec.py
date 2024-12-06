import numpy as np
import py_spec
import sys
from spec_rename import SpecRename
import matplotlib.pyplot as plt

filenames = sys.argv[1:]
fig, axs = plt.subplots(2, 3, figsize=(12, 4))
axs = axs.flatten()

colors = plt.cm.plasma(np.linspace(0, 1, len(filenames)))


colors = ["r", "g", "b", "c", "m", "y", "k"]
print(colors)
colors = colors[:len(filenames)]
for filename, c in zip(filenames, colors):
  try:
    out = py_spec.SPECout(filename) 
  except:
    print("Failed to read ", filename)
  
  for i, ax in enumerate(axs):  
    out.plot_kam_surface(ntheta=128, ax=ax, c=c, label=filename, zeta=i*np.pi/len(axs))
    ax.set_xlabel("s")
    ax.set_ylabel("z")
    ax.set_title(f"Slice $\zeta = {i/len(axs):.2f} \pi$")
    # if i == 0:
    #   ax.legend()
# plt.legend(filenames)
plt.show()

for filename in filenames:
  try:
    out = py_spec.SPECout(filename) 
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    out.plot_kam_surface(ax=axs[0])
    out.plot_poincare(ax=axs[1])
    out.plot_modB(np.linspace(-0.999, 1, 32), np.linspace(0, 2*np.pi, 32), ax=axs[2])
  except:
    print("Failed to read ", filename)
  
plt.show()
