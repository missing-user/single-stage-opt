import numpy as np
import py_spec
import sys
import matplotlib.pyplot as plt

filenames = sys.argv[1:]
fig, axs = plt.subplots(2, 3, figsize=(12, 4))
axs = axs.flatten()
for filename in filenames:
  try:
    out = py_spec.SPECout(filename) 
    out.plot_kam_surface(ntheta=128, zeta=0.0, ax=axs[0], c=None, label=None)
    out.plot_kam_surface(ntheta=128, zeta=0.5 * np.pi, ax=axs[1], c=None, label=None)
    out.plot_kam_surface(ntheta=128, zeta=1  * np.pi, ax=axs[2], c=None, label=None)
    out.plot_kam_surface(ntheta=128, zeta=1.5 * np.pi, ax=axs[3], c=None, label=None)
    out.plot_kam_surface(ntheta=128, zeta=2  * np.pi, ax=axs[4], c=None, label=None)
  except:
    print("Failed to read ", filename)
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
