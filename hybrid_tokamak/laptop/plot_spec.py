import latexplot
latexplot.set_cmap(3)
import matplotlib.pyplot as plt
import numpy as np
import py_spec
import sys
from spec_rename import SpecRename

filenames = sys.argv[1:]
if len(filenames) >1:
  fig, axs = plt.subplots(1, 3, figsize=latexplot.get_size(1, (2,3)), sharex=True, sharey=True)
  axs = axs.flatten()
else:
  fig, axs = plt.subplots(1, 1, figsize=latexplot.get_size(1))
  axs = [axs] * 3

colors = plt.cm.plasma(np.linspace(0, 1, len(filenames)+1))
colors = colors[:len(filenames)]
for filename, c, fi in zip(filenames, colors, range(len(filenames))):
  try:
    out = py_spec.SPECout(filename) 
  except Exception as e: 
    print("Failed to read ", filename, "due to", e)
    continue

  color = c[np.newaxis, :]

  for i, ax in enumerate(axs):  
    label = filename
    label = f"mpol={fi}"

    title = f"Slice $\zeta = {i/(len(axs)-1):.1f} \pi$"
    ns = [1]
    if len(filenames) == 1:
      color = plt.cm.plasma(i/(len(axs)))
      label = title
    out.plot_kam_surface(ntheta=128, ns=ns, ax=ax, c=None, label=label, zeta=i*np.pi/(len(axs)-1), linewidth=1)
    ax.set_title(title)
    if i == len(axs)-1 :
      if plt.isinteractive():
        plt.legend(prop={'size': 6})
      else:
        ax.legend()

    plt.axis("auto")
    ax.label_outer()
    if len(filenames) == 1 and i < len(axs)-1:
      continue
    out.plot_kam_surface(ntheta=128, ns=[2], ax=ax, c="black", label="computational boundary", zeta=0, linewidth=1)
# plt.legend(filenames)
latexplot.savenshow(filename.replace(".sp.h5", "")+"kam")
latexplot.set_cmap(32)
for filename in filenames:
  try:
    out = py_spec.SPECout(filename) 
  except:
    print("Failed to read ", filename)
    continue

  fig, axs = plt.subplots(2, 3, sharex=True,  figsize=latexplot.get_size(1))
  axs = axs.flatten()
  out.plot_kam_surface(ax=axs[0], 
                       c="black",#plt.cm.plasma(0), 
                       linewidth=1)
  # axs[0].set_xlabel("R [m]")
  axs[0].set_ylabel("Z [m]")
  axs[0].set_title("SPEC volume interfaces")
  out.plot_iota(ax=axs[1], c="black")
  axs[1].set_title("iota profile")
  out.plot_modB(np.linspace(-1+1e-6, 1, 64), np.linspace(0, 2*np.pi, 64), lvol=[0], ax=axs[2], cmap=plt.cm.plasma)
  axs[2].set_xlabel("")
  axs[2].set_ylabel("")
  axs[2].set_title("$|B|$")
  def angle2idx(angle):
    return int(np.floor(angle / (2*np.pi) * out.poincare.R.shape[2]))
  
  out.plot_kam_surface(ax=axs[3], 
                       c="black",#plt.cm.plasma(0), 
                       linewidth=1)
  out.plot_poincare(ax=axs[3], s=1, )
  out.plot_poincare(ax=axs[4], s=1, toroidalIdx=angle2idx(0.5*np.pi))
  # axs[4].tick_params('y', labelleft=False)
  out.plot_poincare(ax=axs[5], s=1, toroidalIdx=angle2idx(np.pi))
  for ax in axs:
    ax.label_outer()
  # axs[5].tick_params('y', labelleft=False)
  latexplot.savenshow(filename.replace(".sp.h5", "")+"poincare_iota") 
