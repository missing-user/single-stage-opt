#!/usr/bin/env python
import sys
sys.path.append("./qfb_optimization/")
import latexplot
latexplot.set_cmap(2)
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.io import netcdf
from scipy.interpolate import interp1d
import math

import plotly.express as px


def main(fname):
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

    f = netcdf.netcdf_file(fname, "r", mmap=False)
    nfp = f.variables["nfp"][()]
    nu_plasma = f.variables["nu_plasma"][()]
    nu_middle = f.variables["nu_middle"][()]
    nu_outer = f.variables["nu_outer"][()]
    nv_plasma = f.variables["nv_plasma"][()]
    nv_middle = f.variables["nv_middle"][()]
    nv_outer = f.variables["nv_outer"][()]
    nvl_plasma = f.variables["nvl_plasma"][()]
    nvl_middle = f.variables["nvl_middle"][()]
    nvl_outer = f.variables["nvl_outer"][()]
    u_plasma = f.variables["u_plasma"][()]
    u_middle = f.variables["u_middle"][()]
    u_outer = f.variables["u_outer"][()]
    v_plasma = f.variables["v_plasma"][()]
    v_middle = f.variables["v_middle"][()]
    v_outer = f.variables["v_outer"][()]
    vl_plasma = f.variables["vl_plasma"][()]
    vl_middle = f.variables["vl_middle"][()]
    vl_outer = f.variables["vl_outer"][()]
    r_plasma = f.variables["r_plasma"][()]
    r_middle = f.variables["r_middle"][()]
    r_outer = f.variables["r_outer"][()]
    xm_plasma = f.variables["xm_plasma"][()]
    xm_middle = f.variables["xm_middle"][()]
    xm_outer = f.variables["xm_outer"][()]
    xn_plasma = f.variables["xn_plasma"][()]
    xn_middle = f.variables["xn_middle"][()]
    xn_outer = f.variables["xn_outer"][()]
    mnmax_plasma = f.variables["mnmax_plasma"][()]
    mnmax_middle = f.variables["mnmax_middle"][()]
    mnmax_outer = f.variables["mnmax_outer"][()]
    svd_s_inductance_plasma_outer = f.variables["svd_s_inductance_plasma_outer"][()]
    svd_s_inductance_middle_outer = f.variables["svd_s_inductance_middle_outer"][()]
    svd_s_inductance_plasma_middle = f.variables["svd_s_inductance_plasma_middle"][()]
    svd_s_transferMatrix = f.variables["svd_s_transferMatrix"][()]
    svd_u_transferMatrix_uv = f.variables["svd_u_transferMatrix_uv"][()]
    svd_v_transferMatrix_uv = f.variables["svd_v_transferMatrix_uv"][()]
    svd_u_inductance_plasma_middle_uv = f.variables[
        "svd_u_inductance_plasma_middle_uv"
    ][()]
    svd_v_inductance_plasma_middle_uv = f.variables[
        "svd_v_inductance_plasma_middle_uv"
    ][()]
    n_pseudoinverse_thresholds = f.variables["n_pseudoinverse_thresholds"][()]
    n_singular_vectors_to_save = f.variables["n_singular_vectors_to_save"][()]
    pseudoinverse_thresholds = f.variables["pseudoinverse_thresholds"][()]
    check_orthogonality = False
    try:
        should_be_identity_plasma = f.variables["should_be_identity_plasma"][()]
        should_be_identity_middle = f.variables["should_be_identity_middle"][()]
        should_be_identity_outer = f.variables["should_be_identity_outer"][()]
        check_orthogonality = True
    except:
        pass

    try:
        overlap_plasma = f.variables["overlap_plasma"][()]
        overlap_middle = f.variables["overlap_middle"][()]
        overlap_exists = True
    except:
        overlap_exists = False

    try:
        Bnormal_from_1_over_R_field = f.variables["Bnormal_from_1_over_R_field"][()]
        Bnormal_from_1_over_R_field_uv = f.variables["Bnormal_from_1_over_R_field_uv"][
            ()
        ]
        Bnormal_from_1_over_R_field_inductance = f.variables[
            "Bnormal_from_1_over_R_field_inductance"
        ][()]
        Bnormal_from_1_over_R_field_transfer = f.variables[
            "Bnormal_from_1_over_R_field_transfer"
        ][()]
        one_over_R_exists = True
    except:
        one_over_R_exists = False

    try:
        Bnormal_from_const_v_coils = f.variables["Bnormal_from_const_v_coils"][()]
        Bnormal_from_const_v_coils_uv = f.variables["Bnormal_from_const_v_coils_uv"][()]
        Bnormal_from_const_v_coils_inductance = f.variables[
            "Bnormal_from_const_v_coils_inductance"
        ][()]
        Bnormal_from_const_v_coils_transfer = f.variables[
            "Bnormal_from_const_v_coils_transfer"
        ][()]
        const_v_exists = True
    except:
        const_v_exists = False

    try:
        Bnormal_from_plasma_current = f.variables["Bnormal_from_plasma_current"][()]
        Bnormal_from_plasma_current_uv = f.variables["Bnormal_from_plasma_current_uv"][
            ()
        ]
        Bnormal_from_plasma_current_inductance = f.variables[
            "Bnormal_from_plasma_current_inductance"
        ][()]
        Bnormal_from_plasma_current_transfer = f.variables[
            "Bnormal_from_plasma_current_transfer"
        ][()]
        plasma_current_exists = True
    except:
        plasma_current_exists = False

    print("nu_plasma: ", nu_plasma)
    print("nvl_plasma: ", nvl_plasma)
    print("r_plasma.shape: ", r_plasma.shape)
    print("svd_s_transferMatrix.shape: ", svd_s_transferMatrix.shape)
    print("svd_u_transferMatrix_uv.shape: ", svd_u_transferMatrix_uv.shape)

    f.close()

    ########################################################
    # Plot singular values
    ########################################################

    figureNum = 1
    fig = plt.figure(figureNum)
    fig.patch.set_facecolor("white")

    major_radius = 5.5
    paper_factor = 2 * np.pi**2 * major_radius * 1.25663706127e-6

    plt.plot(
        svd_s_inductance_middle_outer / paper_factor,
        ".m",
        label="Inductance matrix between middle and outer surfaces",
    )
    plt.plot(
        svd_s_inductance_plasma_outer / paper_factor,
        ".r",
        label="Inductance matrix between plasma and outer surfaces",
    )
    plt.plot(
        svd_s_inductance_plasma_middle / paper_factor,
        ".g",
        label="Inductance matrix between plasma and middle surfaces",
    )
    # plt.plot(svd_s_inductance_plasma_middle,'.g',label='Inductance matrix between plasma and control surfaces')
    colors = [
        "k",
        "orange",
        "c",
        "brown",
        "gray",
        "darkred",
        "olive",
        "darkviolet",
        "gold",
        "lawngreen",
    ]
    for i in range(n_pseudoinverse_thresholds):
        plt.plot(
            svd_s_transferMatrix[i, :], ".", color=colors[i], label="Transfer matrix"
        )
    plt.plot(
        svd_s_transferMatrix[i, :],
        ".",
        color=colors[i],
        label="Transfer matrix, thresh=" + str(pseudoinverse_thresholds[i]),
    )
    plt.legend(fontsize="x-small", loc=3)
    plt.title("Singular values (Fig 14.)")
    plt.grid()
    plt.yscale("log")
    ########################################################
    # Fig14
    ########################################################

    figureNum += 1
    fig = plt.figure(figureNum)
    plt.semilogy(
        Bnormal_from_const_v_coils_inductance,
        ".r",
        label="Inductance sequence",
    )
    plt.semilogy(
        Bnormal_from_const_v_coils_inductance / svd_s_inductance_plasma_outer,
        ".g",
        label="Feasibility Sequence Outer",
    )
    plt.semilogy(
        Bnormal_from_const_v_coils_inductance / svd_s_inductance_plasma_middle,
        ".",
        label="Feasibility Sequence Middle",
    )
    plt.legend(fontsize="x-small", loc=3)
    plt.title("(Fig 14.)")
    plt.grid()

    ########################################################
    # For 3D plotting, 'close' the arrays in u and v
    ########################################################

    r_plasma = np.append(r_plasma, r_plasma[[0], :, :], axis=0)
    r_plasma = np.append(r_plasma, r_plasma[:, [0], :], axis=1)
    vl_plasma = np.append(vl_plasma, nfp)

    r_middle = np.append(r_middle, r_middle[[0], :, :], axis=0)
    r_middle = np.append(r_middle, r_middle[:, [0], :], axis=1)
    vl_middle = np.append(vl_middle, nfp)

    r_outer = np.append(r_outer, r_outer[[0], :, :], axis=0)
    r_outer = np.append(r_outer, r_outer[:, [0], :], axis=1)
    vl_outer = np.append(vl_outer, nfp)

    ########################################################
    # Extract cross-sections of the 3 surfaces at several toroidal angles
    ########################################################

    def getCrossSection(rArray, vl_old, v_new):
        vl_old = np.concatenate((vl_old - nfp, vl_old))
        rArray = np.concatenate((rArray, rArray), axis=0)

        print("vl_old shape:", vl_old.shape)
        print("rArray shape:", rArray.shape)

        x = rArray[:, :, 0]
        y = rArray[:, :, 1]
        z = rArray[:, :, 2]
        R = np.sqrt(x**2 + y**2)

        nu = z.shape[1]
        nv_new = len(v_new)
        R_slice = np.zeros([nv_new, nu])
        Z_slice = np.zeros([nv_new, nu])
        for iu in range(nu):
            interpolator = interp1d(vl_old, R[:, iu])
            R_slice[:, iu] = interpolator(v_new)
            interpolator = interp1d(vl_old, z[:, iu])
            Z_slice[:, iu] = interpolator(v_new)

        return R_slice, Z_slice

    v_slices = [0, 0.25, 0.5, 0.75]

    x_plasma = r_plasma[: r_plasma.shape[0] // 2, :, 0]
    y_plasma = r_plasma[: r_plasma.shape[0] // 2, :, 1]
    z_plasma = r_plasma[: r_plasma.shape[0] // 2, :, 2]

    x_middle = r_middle[: r_middle.shape[0] // 2, :, 0]
    y_middle = r_middle[: r_middle.shape[0] // 2, :, 1]
    z_middle = r_middle[: r_middle.shape[0] // 2, :, 2]

    x_outer = r_outer[: r_middle.shape[0] // 2, :, 0]
    y_outer = r_outer[: r_middle.shape[0] // 2, :, 1]
    z_outer = r_outer[: r_middle.shape[0] // 2, :, 2]

    # Create a figure and two subplots for the two surfaces

    figureNum += 1
    fig = plt.figure(figureNum, figsize=(10, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(x_plasma, y_plasma, z_plasma, cmap="plasma")
    ax1.set_title("r_plasma " + str(x_plasma.shape))
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax1.set_zlabel("Z Axis")
    ax1.set_aspect("equal", adjustable="box")

    # First subplot for r_middle
    ax3 = fig.add_subplot(132, projection="3d")
    ax3.plot_surface(x_middle, y_middle, z_middle, cmap="plasma")
    # ax3.plot_surface(-x_plasma, -y_plasma, z_plasma, cmap="plasma")
    ax3.set_title("r_middle")
    ax3.set_xlabel("X Axis")
    ax3.set_ylabel("Y Axis")
    ax3.set_zlabel("Z Axis")
    ax3.set_aspect("equal", adjustable="box")

    # Second subplot for r_outer
    ax2 = fig.add_subplot(133, projection="3d")
    ax2.plot_surface(x_outer, y_outer, z_outer, cmap="plasma")
    ax2.set_title("r_outer")
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")
    ax2.set_aspect("equal", adjustable="box")

    # Show the plot
    plt.tight_layout()
    maximizeWindow()

    R_slice_plasma, Z_slice_plasma = getCrossSection(r_plasma, vl_plasma, v_slices)
    R_slice_middle, Z_slice_middle = getCrossSection(r_middle, vl_middle, v_slices)
    R_slice_outer, Z_slice_outer = getCrossSection(r_outer, vl_outer, v_slices)

    ########################################################
    # Now make plot of surfaces at given toroidal angle
    ########################################################

    figureNum += 1
    fig = plt.figure(figureNum)
    fig.patch.set_facecolor("white")

    numRows = 2
    numCols = 2

    Rmin = R_slice_outer.min()
    Rmax = R_slice_outer.max()
    Zmin = Z_slice_outer.min()
    Zmax = Z_slice_outer.max()

    for whichPlot in range(4):
        plt.subplot(numRows, numCols, whichPlot + 1)
        v = v_slices[whichPlot]
        plt.plot(
            R_slice_outer[whichPlot, :],
            Z_slice_outer[whichPlot, :],
            "b.-",
            label="outer",
        )
        plt.plot(
            R_slice_middle[whichPlot, :],
            Z_slice_middle[whichPlot, :],
            "m.-",
            label="control",
        )
        plt.plot(
            R_slice_plasma[whichPlot, :],
            Z_slice_plasma[whichPlot, :],
            "r.-",
            label="plasma",
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend(fontsize="x-small")
        plt.title("v=" + str(v))
        plt.xlabel("R")
        plt.ylabel("Z")
        plt.xlim([Rmin, Rmax])
        plt.ylim([Zmin, Zmax])

    ########################################################
    # Prepare for plotting singular vectors
    ########################################################

    maxVectorsToPlot = 25
    numVectorsToPlot = min(maxVectorsToPlot, n_singular_vectors_to_save)
    # If more vectors are saved than the number allowed to plot, then plot the last ones saved.
    # plotOffset = n_singular_vectors_to_save - numVectorsToPlot
    plotOffset = 0
    mpl.rc("xtick", labelsize=7)
    mpl.rc("ytick", labelsize=7)
    numCols = int(np.ceil(np.sqrt(numVectorsToPlot)))
    numRows = int(np.ceil(numVectorsToPlot * 1.0 / numCols))

    numContours = 12

    ########################################################
    # Plot singular vectors of the plasma-middle inductance matrix on the plasma surface
    ########################################################

    figureNum += 1
    fig, axs = plt.subplots(numRows, numCols, sharex=True, sharey=True, num=figureNum)
    flat_axs = [x for xs in axs for x in xs]
    for whichPlot in range(numVectorsToPlot):
        plt.sca(flat_axs[whichPlot])
        data = np.reshape(
            svd_u_inductance_plasma_middle_uv[whichPlot + plotOffset, :],
            [nu_plasma, nv_plasma],
            order="F",
        )
        plt.contourf(v_plasma, u_plasma, data, numContours, cmap="jet")
        # plt.colorbar()
        plt.xlabel("v", fontsize="x-small")
        plt.ylabel("u", fontsize="x-small")
        plt.title(
            "Singular vector U "
            + str(whichPlot + plotOffset + 1)
            + f"\ns={svd_s_inductance_plasma_middle[whichPlot + plotOffset]:6f}",
            fontsize="x-small",
        )

        plt.suptitle(
            "Singular vectors of the plasma-to-middle surface inductance matrix. (U vectors = plasma surface)",
            fontsize="small",
        )

    ########################################################
    # Plot singular vectors of the plasma-middle inductance matrix on the middle surface
    ########################################################

    figureNum += 1
    fig = plt.figure(figureNum)
    fig.patch.set_facecolor("white")

    for whichPlot in range(numVectorsToPlot):
        plt.subplot(numRows, numCols, whichPlot + 1)
        data = np.reshape(
            svd_v_inductance_plasma_middle_uv[whichPlot + plotOffset, :],
            [nu_middle, nv_middle],
            order="F",
        )
        plt.contourf(v_middle, u_middle, data, numContours, cmap="jet")
        # plt.colorbar()
        plt.xlabel("v", fontsize="x-small")
        plt.ylabel("u", fontsize="x-small")
        plt.title(
            "Singular vector V "
            + str(whichPlot + plotOffset + 1)
            + "\ns="
            + str(svd_s_inductance_plasma_middle[whichPlot + plotOffset]),
            fontsize="x-small",
        )

        plt.suptitle(
            "Singular vectors of the plasma-to-middle surface inductance matrix. (V vectors = middle surface)",
            fontsize="small",
        )

    ########################################################
    # Plot singular vectors of the transfer matrix on the plasma surface
    ########################################################

    for whichThreshold in range(n_pseudoinverse_thresholds):
        figureNum += 1
        fig = plt.figure(figureNum)
        fig.patch.set_facecolor("white")

        for whichPlot in range(numVectorsToPlot):
            plt.subplot(numRows, numCols, whichPlot + 1)
            data = np.reshape(
                svd_u_transferMatrix_uv[whichThreshold, whichPlot + plotOffset, :],
                [nu_plasma, nv_plasma],
                order="F",
            )
            plt.contourf(v_plasma, u_plasma, data, numContours, cmap="jet")
            # plt.colorbar()
            plt.xlabel("v", fontsize="x-small")
            plt.ylabel("u", fontsize="x-small")
            plt.title(
                "Singular vector U "
                + str(whichPlot + plotOffset + 1)
                + "\ns="
                + str(svd_s_transferMatrix[whichThreshold, whichPlot + plotOffset]),
                fontsize="x-small",
            )

        plt.suptitle(
            "Singular vectors of the transfer matrix. U vectors = plasma surface. (threshold="
            + str(pseudoinverse_thresholds[whichThreshold])
            + ")",
            fontsize="small",
        )

    ########################################################
    # Plot singular vectors of the transfer matrix on the middle surface
    ########################################################

    for whichThreshold in range(n_pseudoinverse_thresholds):
        figureNum += 1
        fig = plt.figure(figureNum)
        fig.patch.set_facecolor("white")

        for whichPlot in range(numVectorsToPlot):
            plt.subplot(numRows, numCols, whichPlot + 1)
            data = np.reshape(
                svd_v_transferMatrix_uv[whichThreshold, whichPlot + plotOffset, :],
                [nu_middle, nv_middle],
                order="F",
            )
            plt.contourf(v_middle, u_middle, data, numContours, cmap="jet")
            # plt.colorbar()
            plt.xlabel("v", fontsize="x-small")
            plt.ylabel("u", fontsize="x-small")
            plt.title(
                "Singular vector V "
                + str(whichPlot + plotOffset + 1)
                + "\ns="
                + str(svd_s_transferMatrix[whichThreshold, whichPlot + plotOffset]),
                fontsize="x-small",
            )

        plt.suptitle(
            "Singular vectors of the transfer matrix. V vectors = middle surface. (threshold="
            + str(pseudoinverse_thresholds[whichThreshold])
            + ")",
            fontsize="small",
        )

    #######################################################
    # Plot quantities related to 1/R field
    ########################################################

    if one_over_R_exists or const_v_exists or plasma_current_exists:

        figureNum += 1
        fig = plt.figure(figureNum)
        fig.patch.set_facecolor("white")

        numRows = 2
        numCols = 3
        titleFontSize = 10

        if one_over_R_exists:
            plt.subplot(numRows, numCols, 1)
            plt.contourf(
                v_plasma,
                u_plasma,
                Bnormal_from_1_over_R_field_uv.transpose(),
                numContours,
            )
            plt.xlabel("v", fontsize="x-small")
            plt.ylabel("u", fontsize="x-small")
            # plt.imshow(,interpolation='none')
            ##plt.gca().xaxis.tick_top()
            plt.colorbar()
            plt.title("1/R toroidal field\n(component normal to plasma surface)")
            # plt.xlabel('iu')
            # plt.ylabel('iv')

            plt.subplot(numRows, numCols, 4)
            plt.semilogy(
                abs(Bnormal_from_1_over_R_field), "x-", label="Basis functions"
            )
            plt.semilogy(
                abs(Bnormal_from_1_over_R_field_inductance),
                "x-",
                label="Left-singular vectors of inductance matrix",
            )
            plt.semilogy(
                abs(Bnormal_from_1_over_R_field_transfer),
                "x-",
                label="Left-singular vectors of transfer matrix",
            )
            plt.title(
                "abs (Bnormal_from_1_over_R_field) on the plasma surface",
                fontsize=titleFontSize,
            )
            plt.xlabel("index")
            plt.legend(fontsize=8)

        if const_v_exists:
            plt.subplot(numRows, numCols, 2)
            plt.contourf(
                v_plasma,
                u_plasma,
                Bnormal_from_const_v_coils_uv.transpose(),
                numContours,
            )
            plt.xlabel("v", fontsize="x-small")
            plt.ylabel("u", fontsize="x-small")
            # plt.imshow(,interpolation='none')
            ##plt.gca().xaxis.tick_top()
            plt.colorbar()
            plt.title(
                "B field due to constant-v coils\n(component normal to plasma surface)"
            )
            # plt.xlabel('iu')
            # plt.ylabel('iv')

            plt.subplot(numRows, numCols, 5)
            plt.semilogy(abs(Bnormal_from_const_v_coils), "+-", label="Basis functions")
            plt.semilogy(
                abs(Bnormal_from_const_v_coils_inductance),
                "+-",
                label="Left-singular vectors of inductance matrix",
            )
            plt.semilogy(
                abs(Bnormal_from_const_v_coils_transfer),
                "+-",
                label="Left-singular vectors of transfer matrix",
            )
            plt.title(
                "abs (Bnormal_from_const_v_coils) on the plasma surface",
                fontsize=titleFontSize,
            )
            plt.xlabel("index")
            plt.legend(fontsize=8)

        if plasma_current_exists:
            plt.subplot(numRows, numCols, 3)
            plt.contourf(
                v_plasma,
                u_plasma,
                Bnormal_from_plasma_current_uv.transpose(),
                numContours,
            )
            plt.xlabel("v", fontsize="x-small")
            plt.ylabel("u", fontsize="x-small")
            # plt.imshow(,interpolation='none')
            ##plt.gca().xaxis.tick_top()
            plt.colorbar()
            plt.title(
                "B field due to plasma current\n(component normal to plasma surface)"
            )
            # plt.xlabel('iu')
            # plt.ylabel('iv')

            plt.subplot(numRows, numCols, 6)
            try:
                plt.semilogy(
                    abs(Bnormal_from_plasma_current), "+-", label="Basis functions"
                )
                plt.semilogy(
                    abs(Bnormal_from_plasma_current_inductance),
                    "+-",
                    label="Left-singular vectors of inductance matrix",
                )
                plt.semilogy(
                    abs(Bnormal_from_plasma_current_transfer),
                    "+-",
                    label="Left-singular vectors of transfer matrix",
                )
            except:
                # May give an error if values are all 0
                pass

            plt.title(
                "abs (Bnormal_from_plasma_current) on the plasma surface",
                fontsize=titleFontSize,
            )
            plt.xlabel("index")
            plt.legend(fontsize=8)

        maximizeWindow()

    plt.show()

    ########################################################
    # Now make the publication plots
    
    def fit_exponential_rate(sequence):
        x = np.linspace(0, 1, len(sequence))

        # log-linear fit 
        fit = np.polyfit(x, np.log(sequence), 1)
        a_initial = np.exp(fit[1]) #sequence[0]
        b_initial = fit[0]

        # Exponential curve fit using log linear as initial guess
        # Define the exponential function for fitting
        def exp_model(x, a, b, c):
            return a * np.exp(b * x) + c
        
        return exp_model(x, a_initial, b_initial, 0)

    # latexplot.figure()

    fig, axd = plt.subplot_mosaic(
        [[f"svd{i}" for i in range(6)],
         [ "BdotN","BdotN","BdotN","cross","cross","cross",],
        #  [ "BdotN","BdotN",  "cross","cross",],
        #  [ "BdotN","BdotN",  "cross","cross",],
        #  [ "BdotN","BdotN",  "cross","cross",],
         [ "Se",]*6,
        ],
        layout="constrained",
        # width_ratios=[5,1,5],
        height_ratios=[0.6,1.5,2],
        figsize=latexplot.get_size(subplots=(3,2))
    )
    plt.sca(axd["Se"])
    plt.scatter(
        np.arange(len(Bnormal_from_1_over_R_field_inductance)),
        abs(Bnormal_from_1_over_R_field_inductance),
        # ".",
        s=1,
        label=r"$S_{e, inductance}$",
    )
    plt.semilogy(
        fit_exponential_rate(abs(Bnormal_from_1_over_R_field_inductance)),
        "-",
        label=r"$S_{e, inductance}$ fit",
    )


    plt.scatter(
        np.arange(len(Bnormal_from_1_over_R_field_transfer)),
        abs(Bnormal_from_1_over_R_field_transfer),
        # ".",
        s=1,
        label=r"$S_{e, transfer}$",
    )
    plt.semilogy(
        fit_exponential_rate(abs(Bnormal_from_1_over_R_field_transfer)),
        "-",
        label=r"$S_{e, transfer}$ fit",
    )
    plt.grid()
    plt.legend()
    plt.xlabel("Index of the Singular Vector $U_i$")
    plt.ylabel(r"$S_e$ value")
    # plt.ylabel(r"Efficiency Sequence $S_e = U^T \cdot \vec{B}_{ext, n}$")
    plt.title(r"Efficiency Sequence $S_e$")


    # Crossection plot
    latexplot.set_cmap(3)
    plt.sca(axd["cross"])
    whichPlot = 0
    plt.plot(
        R_slice_outer[whichPlot, :],
        Z_slice_outer[whichPlot, :],
        label="$\mathcal{D}$ computational",
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    )
    plt.plot(
        R_slice_middle[whichPlot, :],
        Z_slice_middle[whichPlot, :],
        label="$\mathcal{M}$ middle",
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    )
    plt.plot(
        R_slice_plasma[whichPlot, :],
        Z_slice_plasma[whichPlot, :],
        label="$\mathcal{S}$ plasma",
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Cross-Sections $\phi=0$") 
    plt.xlabel("R")
    plt.ylabel("Z")
    # plt.xlim([Rmin, Rmax])
    plt.xlim([-0.5,1.7])
    plt.legend()#fontsize="x-small")


    for svdi in range(6):
        plt.sca(axd[f"svd{svdi}"])
        
        data = np.reshape(
            svd_u_transferMatrix_uv[0, svdi, :],
            [nu_plasma, nv_plasma],
            order="F",
        )
        plt.imshow(data)
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        # plt.colorbar()
        plt.title(
            f"$U_{svdi+1}$"
            # + f"\ns={svd_s_transferMatrix[0,svdi]:3f}",
            # fontsize="x-small",
        )

        
    plt.sca(axd["BdotN"])
    plt.imshow(Bnormal_from_1_over_R_field_uv, aspect='auto')
    # plt.colorbar()
    plt.title(r"$\vec{B}_{ext} \cdot \vec{n}$ on surface $\mathcal{D}$")
    plt.xlabel(r"toroidal angle $\phi$")
    plt.ylabel(r"poloidal angle $\theta$")
    plt.gcf().get_layout_engine().set(w_pad=0.01)
    latexplot.savenshow("plots/bdistrib_examples/bdistrib_plot")
    

if __name__ == "__main__":

    print("usage: python bdistribPlot.py bdistrib_out.XXX.nc")

    import sys

    if len(sys.argv) != 2:
        print("Error! You must specify 1 argument: the bdistrib_out.XXX.nc file.")
        exit(1)
    main(sys.argv[1])
