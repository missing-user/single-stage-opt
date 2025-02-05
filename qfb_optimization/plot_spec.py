import latexplot
latexplot.set_cmap(3)
import matplotlib.pyplot as plt
import numpy as np
import py_spec
import sys
from spec_rename import SpecRename

plot_boundary = True
plot_boundary = False
filenames = sys.argv[1:]
if len(filenames) >1:
  fig, axs = plt.subplots(1, 3, figsize=latexplot.get_size(1, (2,3)), sharex=True, sharey=True)
  axs = axs.flatten()
else:
  fig, axs = plt.subplots(1, 1, figsize=latexplot.get_size(1))
  axs = [axs] * 3


fig, axs = plt.subplots(2, 3, figsize=latexplot.get_size(1, (4,3)), sharex=True, sharey=True)
axs_bot = axs[1,:].flatten()
axs = axs[0,:].flatten()

import plot_vmec
fnames = [
  "fixb_12-31-16-42-24/mpol1/input.rot_ellipse_000_000000", 
  "fixb_12-31-16-42-24/mpol1/input.rot_ellipse_000_000051", 
  "fixb_12-31-16-42-24/mpol2/input.rot_ellipse_000_000192", 
  "fixb_12-31-16-42-24/mpol3/input.rot_ellipse_000_000456", 
  "fixb_12-31-16-42-24/mpol4/input.rot_ellipse_000_000948",  
  "fixb_12-31-16-42-24/mpol5/input.rot_ellipse_000_001330",
]
from simsopt import mhd
vmecs = [mhd.Vmec(filename) for filename in fnames]
plot_vmec.compare_crosssections(vmecs, axs=axs_bot)

if len(filenames) == 1:
  plot_boundary = True

def dof_from_mpol(mpol):
  return mpol*(mpol*2+1)+mpol 

colors = plt.cm.plasma(np.linspace(0, 1, len(filenames), endpoint=False))
colors = np.array(list(reversed(colors)))
# colors = np.array([plt.cm.tab10(i) for i in range(len(filenames))])
for filename, c, fi in zip(filenames, colors, range(len(filenames))):
  try:
    out = py_spec.SPECout(filename) 
  except Exception as e: 
    print("Failed to read ", filename, "due to", e)
    continue

  color = c[np.newaxis, :]

  for i, ax in enumerate(axs):  
    label = filename
    label = f"M=N={fi}"
    # label = f"{dof_from_mpol(fi)} DOFs"

    title = f"QFB $\phi = {i/(len(axs)-1):.1f} \pi" +"/ n_{fp}$"
    ns = [1]
    if len(filenames) == 1:
      color = plt.cm.plasma(i/(len(axs)))
      label = title
    if i != 0:
      label = None

    out.plot_kam_surface(ntheta=128, ns=ns, ax=ax, c=color, label=label, zeta=i*np.pi/(len(axs)-1), linewidth=1)
    ax.set_title(title)
    if fi == len(filenames)-1 and plot_boundary:
      out.plot_kam_surface(ntheta=128, ns=[2], ax=ax, c="black", label="$\mathcal{D}$", zeta=0, linewidth=1)
    if i == 0 and fi == len(filenames)-1 and False:
      if plt.isinteractive():
        fig.legend(prop={'size': 6}, loc="outside lower right") 
      else:
        fig.legend(loc="outside lower right")

    ax.axis("auto")
    ax.label_outer()
    if len(filenames) == 1 and i < len(axs)-1:
      continue

if len(filenames) == 1:
  plt.axis("equal") 
else:
  plt.axis("auto")
# plt.legend(filenames)
latexplot.savenshow(filename.replace(".sp.h5", "")+"kam")
print("Saved", filename.replace(".sp.h5", "")+"kam")
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
