# %%
import sys
sys.path.append("../../hybrid_tokamak/laptop/")
import latexplot
latexplot.set_cmap(3)
import numpy as np
import matplotlib.pyplot as plt
import simsopt
from simsopt import mhd
from simsopt import geo
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from simsopt.configs import get_QUASR_data
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry


import pandas as pd
df:pd.DataFrame = pd.read_pickle("../../quasr_coil_check/QUASR_db/QUASR_08072024.pkl")

"""
Sample 200 configurations with 4, 6, and 24 coils and plot the L_{filament} to LgradB correlations. 
They cluster together by coil number.
"""


# Filter df by constant number of coils
# df = df[df["nc_per_hp"] * df["nfp"] == 6]
# %%
df["$n_{coils}$"] = 2*(df["nc_per_hp"] * df["nfp"]).astype(int)
# Pick the top 5 bins
most_common = df["$n_{coils}$"].value_counts()
# df = df[df["$n_{coils}$"].isin(most_common.index[0:5])]
# df = df[df["$n_{coils}$"].isin([4, 6, 24])]

df = pd.concat([
    df[df["$n_{coils}$"]==4 ].sample(n=200, replace=False, random_state=42),
    df[df["$n_{coils}$"]==6 ].sample(n=200, replace=False, random_state=42),
    df[df["$n_{coils}$"]==24].sample(n=200, replace=False, random_state=42),
])
# df = df.sample(n=500, replace=False, random_state=42)



ids = df["ID"].tolist()


def get_QUASR_data_sync(idx, style='quasr-style'):
    # Your original synchronous function
    data = get_QUASR_data(idx, style)
    return idx, data[0], data[1]

async def fetch_data(idx, loop, executor):
    idx, surfs, coils = await loop.run_in_executor(executor, get_QUASR_data_sync, idx)
    return idx, surfs, coils
    # try:
    # # Network error
    # except ValueError as e:
    #     print(f"Network error with {idx}: {e}")
    #     return idx, None, None

# loop = asyncio.get_event_loop()
# executor = ThreadPoolExecutor()
# tasks = [fetch_data(idx, loop, executor) for idx in ids]
# results = asyncio.gather(*tasks)
results = map(get_QUASR_data_sync, ids)
results = list(filter(lambda x: x[1] is not None, results))
# loop.stop()
# executor.shutdown()

# %%
def normalize_scale(surface, constant_minor=True):
  # Scaling factor for either constant minor or major radius
  if constant_minor:
    scaling = 1.704 / surface.minor_radius()
  else: 
    scaling = 1.0 / surface.major_radius()
  return scaling

# %%
coil_surf_dist = {}

def coil_surf_distance(curves, lcfs) -> np.ndarray:
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    distances = [np.min(cdist(pointcloud1, c.gamma()), axis=0) for c in curves]
    return np.array(distances).T


def compute_coil_surf_dist(simsopt_filename):
    surfaces, coils = simsopt.load(simsopt_filename)
    lcfs = surfaces[-1].to_RZFourier()

    curves = [c.curve for c in coils]
    return coil_surf_distance(curves, lcfs)

for idx, surfs, coils in results:
  coil_surf_dist[idx] = coil_surf_distance([c.curve for c in coils], surfs[-1]) * normalize_scale(surfs[-1]) 
df["QUASR coil distance"] = df["ID"].map({int(k): np.min(v) for k, v in coil_surf_dist.items()})

# %%

from joblib import Memory
location = './.cachedir'
memory = Memory(location, verbose=0)

@memory.cache(ignore=["vmec"])
def vmec_lgradbsc(vmec, idx):
    try:
        vmec.run()
        s = [1]
        ntheta = 64
        nphi = 64
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi / vmec.boundary.nfp, nphi, endpoint=False)
        data = vmec_compute_geometry(vmec, s, theta, phi)
        return np.min(data.L_grad_B)
    except Exception as e:
        print(f"VMEC didn't converge! Error with {idx}: {str(e)}")
    return 0.0


# %%
vmec = mhd.Vmec(verbose=False)
LgradB_keyed = {}
for idx, surfs, coils in results:
    s : geo.SurfaceRZFourier = surfs[-1].to_RZFourier().copy()

    # If the configurations are all scaled to the same major radius instead of minor radius, the results that follow are not qualitatively different
    scaling = normalize_scale(s)
    s.rc_array *= scaling
    s.zs_array *= scaling

    vmec.boundary = s
    vmec.indata.nfp = vmec.boundary.nfp
    # print(idx)
    LgradB_keyed[idx] = vmec_lgradbsc(vmec, idx)
    print(idx)

df["$L^*_{\\nabla \\vec{B}}$"] = df["ID"].map(LgradB_keyed)
df.to_pickle("coil_distances.pkl")
df = df[df["$L^*_{\\nabla \\vec{B}}$"] > 0.0].dropna()
# %%
# df.plot(x="$L^*_{\\nabla \\vec{B}}$", kind="scatter", y="QUASR coil distance", c="$n_{coils}$")

dff = df[df["$L^*_{\\nabla \\vec{B}}$"] > 0.0].dropna()
# dff = dff[(dff["$n_{coils}$"] == 24) | (dff["$n_{coils}$"] == 6) ]

# Scatter plot
latexplot.figure()

# Add linear fits for each unique `$n_{coils}$`
for n_coils, group in dff.groupby("$n_{coils}$"):
    x = group["QUASR coil distance"]
    y = group["$L^*_{\\nabla \\vec{B}}$"]
    # Perform linear fit
    coeffs = np.polyfit(x, y, 1)  # First-order polynomial
    print(coeffs)
    fit_line = np.poly1d(coeffs)
    
    # Plot the fit line
    plt.scatter(x, y, label=f"data $n_{{coils}} = {n_coils}$", s=4)
    plt.plot(x, fit_line(x), label=f"fit $n_{{coils}} = {n_coils}$")

# Add legend, labels, and color bar
plt.xlabel("$L_{filament}$")
plt.ylabel("$L^*_{\\nabla \\vec{B}}$")
plt.title("Clustering of $L_{filament}$ behavior by $n_{coils}$")
plt.legend()

latexplot.savenshow("coil_distances")
