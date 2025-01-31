#!/usr/bin/env python

from scipy.stats import linregress
import inspect
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import inspect
import os
from scipy.io import netcdf
import sys

myname = inspect.getfile(inspect.currentframe())
print("This is " + myname)
print("Usage:")
print("  " + myname + " <List of 1 or more bdistrib_out.XXX.nc files>")
print()

if len(sys.argv) < 2:
    print(
        "Error! You must list at least one bdistrib_out.XXX.nc file as a command-line argument."
    )
    exit(1)


def maximizeWindow():
    # Maximize window. The command for this depends on the backend.
    mng = plt.get_current_fig_manager()
    try:
        mng.resize(*mng.window.maxsize())
    except AttributeError:
        try:
            mng.window.showMaximized()
        except AttributeError:
            pass


many_properties = {
    "svd_s_transferMatrix": [],
    "svd_s_inductance_plasma_middle": [],
    # "svd_s_inductance_plasma_outer": [],
    # "Bnormal_from_1_over_R_field": [],
    "Bnormal_from_1_over_R_field_inductance": [],
    "Bnormal_from_1_over_R_field_transfer": [],
}

dataNames = []

with open(sys.argv[1]) as f:
    import json

    print(f, sys.argv[1])
    regcoil_distance_dict = json.load(f)
first_file = 2

for whichFile in range(first_file, len(sys.argv)):
    filename = sys.argv[whichFile]
    f = netcdf.netcdf_file(filename, "r", mmap=False)
    for key in many_properties.keys():
        tmpval = f.variables[key][()]
        if tmpval.shape[0] == 1:
            tmpval = tmpval.flatten()

        many_properties[key].append(tmpval)
    dataNames.append(
        # + " (thresh=" + str(pseudoinverse_thresholds[whichThreshold]) + ")"
        filename
    )

    f.close()

    print("Read data from file " + filename)


# Sort the sequences by regcoil distance
distances = np.array(
    [
        regcoil_distance_dict[name.split(
            "bdistrib_out.")[-1].replace(".nc", ".json")]
        for name in dataNames
    ]
)
print(distances)
dataNames = [
    f"{name}   dist={distance:.3f}" for name, distance in zip(dataNames, distances)
]
distance_sorted_ids = np.argsort(distances)


# Make plot for efficiency decay rate vs REGCOIL distance


def fit_exponential_rate(sequence):
    x = np.linspace(0, 1, len(sequence))
    fit = np.polyfit(x, np.log(sequence), 1)
    return fit[0]


plt.figure()
for prop, propval in many_properties.items():
    if prop.count("svd") == 0:
        decay_rates = [fit_exponential_rate(np.abs(seq)) for seq in propval]
        plt.plot(distances, decay_rates, ".", label=prop)

        decay_rates = np.asarray(decay_rates)
        distances = np.asarray(distances)

        finite_mask = np.isfinite(distances)
        decay_rates = decay_rates[finite_mask]
        distances2 = distances[finite_mask]

        reg = linregress(distances2, decay_rates)
        plt.axline(
            xy1=(0, reg.intercept),
            slope=reg.slope,
            color="k",
            label=f"Linear fit {prop}: $R^2$ = {reg.rvalue**2:.3f}",
        )
plt.xlabel(r"$L_{REGCOIL}$")
plt.ylabel("Rate of increase $\\gamma$")
plt.legend()

for prop, propval in many_properties.items():
    # Create a Plotly figure
    fig = go.Figure()

    # Loop over sorted IDs and create the traces
    for idx, i in enumerate(distance_sorted_ids):
        fig.add_trace(
            go.Scatter(
                y=propval[i].astype(np.float64),
                mode="markers",
                name=dataNames[i],
            )
        )

    # Add title and grid
    fig.update_layout(
        title=prop,
        showlegend=True,
        legend=dict(font=dict(size=10), x=0.01, y=0.99),
        xaxis_title="Index",
        yaxis_title="Value (log scale)",
        yaxis_type="log",
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
    )

    # Add text on the plot for additional details
    titleString = (
        "Plot generated by "
        + os.path.abspath(inspect.getfile(inspect.currentframe()))
        + "\nRun in "
        + os.getcwd()
    )

    fig.add_annotation(
        text=titleString,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=10),
        align="center",
    )

    # Display the plot
    fig.show()


##########################################################
# Make plot for inductance matrix
##########################################################
for prop, propval in many_properties.items():
    fig = plt.figure()
    fig.patch.set_facecolor("white")

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.turbo(np.linspace(0, 1, len(sys.argv) - first_file))
    )
    print(np.ptp(distances[distances < 1e6]))
    colors = plt.cm.viridis(
        (distances - np.min(distances)) / np.ptp(distances[distances < 1e6])
    )

    for i in distance_sorted_ids:  # range(len(dataNames)):
        plt.semilogy(propval[i], ".", label=dataNames[i], c=colors[i])
    plt.title(prop)
    plt.legend(frameon=False, prop=dict(size="x-small"), loc=3)
    plt.grid(True)

    titleString = (
        "Plot generated by "
        + os.path.abspath(inspect.getfile(inspect.currentframe()))
        + "\nRun in "
        + os.getcwd()
    )
    ax = fig.add_axes((0, 0, 1, 1), frameon=False)
    ax.text(
        0.5, 0.99, titleString, horizontalalignment="center", verticalalignment="top"
    )

    maximizeWindow()


plt.show()
