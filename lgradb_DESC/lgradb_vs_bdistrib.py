import desc.io
import desc.plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import simsopt
import desc.grid

import json
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from scipy.io import netcdf
import scipy.ndimage.filters
import warnings
from pathlib import Path

MAX_SVD_VALS = 40
SINGLE_STAGE_PATH = Path.home() / "single-stage-opt/"


def coil_surf_distance(curves, lcfs) -> np.ndarray:
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    distances = [np.min(cdist(pointcloud1, c.gamma()), axis=0) for c in curves]
    return np.array(distances).T


def compute_coil_surf_dist(simsopt_filename):
    surfaces, coils = simsopt.load(simsopt_filename)
    lcfs = surfaces[-1].to_RZFourier()

    curves = [c.curve for c in coils]
    return coil_surf_distance(curves, lcfs)


def fit_exponential_rate(sequence, title):
    x = np.linspace(0, 1, len(sequence))
    # plt.semilogy(x, sequence)
    fit = np.polyfit(x, np.log(sequence), 1)
    # plt.semilogy(x, np.exp(np.polyval(fit, x)))
    # plt.title(title)
    # plt.show()
    return fit


def compare_bdistrib(simsopt_name):
    filepath = f"{SINGLE_STAGE_PATH}/replicate_lgradb/tmp/dist05/bdistrib_out.{simsopt_name.replace('.json','.nc')}"
    # Some fits of the exponential decay, take a look at compareSingularValuesPlot.py and bdistrib_util.py

    bdistrib_variables = [
        "Bnormal_from_const_v_coils_inductance",
        "Bnormal_from_const_v_coils_transfer",
        # "svd_s_transferMatrix",
        # "svd_s_inductance_plasma_middle",
    ]
    fit_types = [
        "log_linear",
        #  "value",
        #  "windowed_upper_bound"
    ]
    fits = {}
    with netcdf.netcdf_file(filepath, "r", mmap=False) as f:
        for key in bdistrib_variables:
            vararray = f.variables[key][()].flatten()
            vararray = np.abs(vararray)  # [:MAX_SVD_VALS]
            for fit_type in fit_types:
                if fit_type == "value":
                    fit_val = np.max(np.log(vararray))
                elif fit_type == "log_linear":
                    # Only take the 0 element (slope) of the fit
                    fit_val = fit_exponential_rate(vararray, key)[0]
                else:
                    windowed_array = scipy.ndimage.filters.maximum_filter1d(
                        vararray, size=16
                    )
                    fit_val = fit_exponential_rate(windowed_array, key)[0]
                fits[key + fit_type] = fit_val
    print(list(fits.keys()))
    return fits


def vs_plot(x_data, y_data, labels=None):
    x_vals, x_label = x_data
    y_vals, y_label = y_data
    title = x_label + " vs " + y_label

    if len(np.shape(y_vals)) >= 2:
        y_vals = np.array(y_vals).T
    elif len(np.shape(y_vals)) == 1:
        y_vals = np.reshape(y_vals, (1,) + np.shape(y_vals))

    # Linear fit
    # TODO this Fails because some values are inf!!
    for i, y in enumerate(y_vals):
        plt.scatter(x_vals, y, label=title if labels is None else labels[i])
        reg = linregress(x_vals, y)
        plt.axline(
            xy1=(0, reg.intercept),
            slope=reg.slope,
            color="k",
            label=f"Linear fit {i}: $R^2$ = {reg.rvalue:.2f}",
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if np.min(y_vals) >= 0:
        plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.grid(True)
    plt.legend()


if __name__ == "__main__":
    regcoil_plot = True
    bdistrib_plot = True

    try:
        with open("all_results.json") as f:
            regcoil_distances = json.load(f)
            for key in regcoil_distances.keys():
                regcoil_distances[key] /= 5
    except Exception as e:
        warnings.warn(e.__str__())
        regcoil_plot = False

    LgradB_keyed = {}
    coil_surf_dist = {}

    desc_outputs = list(
        filter(lambda x: x.endswith("_output.h5"), os.listdir()))
    plotgrid = desc.grid.LinearGrid(theta=128, zeta=128)
    for filename in desc_outputs:
        eq_fam = desc.io.load(filename)
        eq = eq_fam[-1]

        # Compute LgradB
        computed = eq.compute(
            [
                "|B|",
                "grad(B)",
                "L_grad(B)",
                # "<|B|>",
                # "<|B|>_vol",
                # "<|B|>_axis",
            ]
        )

        #     print(
        #         f"|B|_max {np.max(computed['|B|']):.3f} <|B|>_mean\
        # {np.mean(computed['<|B|>']):.3f} <|B|>_vol {computed['<|B|>_vol']:.3f}\
        # <|B|>_axis {computed['<|B|>_axis']:.3f}"
        #     )

        LgradBs = computed["L_grad(B)"]
        LgradBnucs = (
            np.sqrt(2)
            * computed["|B|"]
            / np.linalg.norm(computed["grad(B)"], ord="nuc", axis=(1, 2))
        )
        LgradB2s = (
            np.sqrt(2)
            * computed["|B|"]
            / np.linalg.norm(computed["grad(B)"], ord=2, axis=(1, 2))
        )
        LgradB_lcfs = eq.compute(["L_grad(B)"], plotgrid)["L_grad(B)"]
        LgradB = np.min(LgradB_lcfs)
        print(np.min(LgradB_lcfs), "<", np.min(LgradBs))

        # REGCOIL distance
        simsopt_name = filename.replace("input.", "serial").replace(
            "_output.h5", ".json"
        )
        LgradB_keyed[simsopt_name] = np.array(
            [LgradB, np.min(LgradB2s), np.min(LgradBnucs)]
        )

        # The distances here were verified with the QUASR database GUI and are correct.
        simsopt_path = f"{SINGLE_STAGE_PATH}/quasr_coil_check/QUASR_db/simsopt_serials/{simsopt_name[6:10]}/{simsopt_name}"
        coil_surf_dist[simsopt_name] = compute_coil_surf_dist(simsopt_path)
        print(
            simsopt_name,
            "has minimum filament coil distance",
            np.min(coil_surf_dist[simsopt_name]),
        )

        # Only plot all individual equilibria for small datasets
        if len(desc_outputs) <= 10 or LgradB < 0.05:
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            desc.plotting.plot_boundary(eq, ax=ax[0, 0])

            LgradB_inv = plotgrid.meshgrid_reshape(1 / LgradB_lcfs, "rtz")[0]
            ax[0, 1].imshow(LgradB_inv)
            # ax[0, 1].colorbar()

            # desc.plotting.plot_2d(eq, "L_grad(B)", ax=ax[0, 1], cmap="jet_r")
            # desc.plotting.plot_2d(eq, "L_grad(B)", ax=ax[0, 1], cmap="jet_r")

            if regcoil_plot:
                ax[1, 1].hlines(
                    LgradB,
                    0,
                    1,
                    linestyles="dashed",
                )
                ax[1, 1].hlines(
                    regcoil_distances[simsopt_name],
                    0,
                    1,
                )
                ax[1, 1].set_ylim(bottom=0)
                ax[1, 1].legend(
                    ["$L^*_{\\nabla B}$", "Coil winding surf. dist."])

            # QUASR Filament coil distance
            ax[1, 0].plot(coil_surf_dist[simsopt_name], label="coil")
            ax[1, 0].hlines(
                LgradB, 0, coil_surf_dist[simsopt_name].shape[0], linestyles="dashed"
            )
            ax[1, 0].set_title("Filament coil distance")
            ax[1, 0].legend()
            ax[1, 0].set_ylim(bottom=0)

            fig.suptitle(filename)
            fig.show()

    #########################

    # Extract filenames and corresponding values for plotting
    filenames = list(LgradB_keyed.keys())
    if regcoil_plot:
        regcoil_vals = ([regcoil_distances[f]
                        for f in filenames], "REGCOIL distance")
    LgradB_vals = ([LgradB_keyed[f] for f in filenames], "$L^*_{\\nabla B}$")
    if bdistrib_plot:
        try:
            Bdistrib_vals = (
                [list(compare_bdistrib(f).values()) for f in filenames],
                "efficient fields seqence",
            )
        except Exception as e:
            warnings.warn(e.__str__())
            bdistrib_plot = False

    coil_min_vals = (
        [np.min(coil_surf_dist[f]) for f in filenames],
        "QUASR coil distance",
    )

    #########################
    if regcoil_plot:
        plt.figure()
        vs_plot(regcoil_vals, LgradB_vals)
        plt.figure()
        vs_plot(regcoil_vals, coil_min_vals)
    plt.figure()
    vs_plot(coil_min_vals, LgradB_vals)

    if bdistrib_plot:
        plt.figure()
        vs_plot(
            coil_min_vals,
            Bdistrib_vals,
            [
                "Bnormal_from_const_v_coils_inductancelog_linear",
                # "Bnormal_from_1_over_R_field_inductancevalue",
                # "Bnormal_from_1_over_R_field_inductancewindowed_upper_bound",
                "Bnormal_from_const_v_coils_transferlog_linear",
                # "Bnormal_from_1_over_R_field_transfervalue",
                # "Bnormal_from_1_over_R_field_transferwindowed_upper_bound",
                # "svd_s_transferMatrixlog_linear",
                # "svd_s_transferMatrixvalue",
                # "svd_s_transferMatrixwindowed_upper_bound",
                # "svd_s_inductance_plasma_middlelog_linear",
                # "svd_s_inductance_plasma_middlevalue",
                # "svd_s_inductance_plasma_middlewindowed_upper_bound",
            ],
        )

        if regcoil_plot:
            plt.figure()
            vs_plot(
                regcoil_vals,
                Bdistrib_vals,
                [
                    "Bnormal_from_const_v_coils_inductancelog_linear",
                    "Bnormal_from_const_v_coils_transferlog_linear",
                ],
            )

    # Plot over filenames: regcoil_distances, filament coil distance, L*_{\nabla B}
    plt.figure(figsize=(12, 8))

    if regcoil_plot:
        plt.plot(filenames, regcoil_vals[0], marker="o", label=regcoil_vals[1])
    plt.plot(filenames, coil_min_vals[0], marker="s", label=coil_min_vals[1])
    plt.plot(filenames, LgradB_vals[0], marker="^", label=LgradB_vals[1])
    plt.xlabel("Filenames")
    plt.ylabel("Values")
    plt.title("Comparison of different metrics")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
