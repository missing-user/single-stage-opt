# plotly version of the regcoilPlot plotting script from
# https://github.com/landreman/regcoil.
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.io import netcdf_file
from scipy.interpolate import interp1d


def load_netcdf_data(filename):
    f = netcdf_file(filename, "r", mmap=False)

    # NETCdf internally uses different endianness, maybe even Fortran
    # ordering. np.ascontiguousarray fixes that
    data = {
        "nfp": int(f.variables["nfp"][()]),
        "ntheta_plasma": int(f.variables["ntheta_plasma"][()]),
        "ntheta_coil": int(f.variables["ntheta_coil"][()]),
        "nzeta_plasma": int(f.variables["nzeta_plasma"][()]),
        "nzeta_coil": int(f.variables["nzeta_coil"][()]),
        "nzetal_plasma": int(f.variables["nzetal_plasma"][()]),
        "nzetal_coil": int(f.variables["nzetal_coil"][()]),
        "theta_plasma": np.ascontiguousarray(f.variables["theta_plasma"][()]).astype(
            float
        ),
        "theta_coil": np.ascontiguousarray(f.variables["theta_coil"][()]).astype(float),
        "zeta_plasma": np.ascontiguousarray(f.variables["zeta_plasma"][()]).astype(
            float
        ),
        "zeta_coil": np.ascontiguousarray(f.variables["zeta_coil"][()]).astype(float),
        "zetal_plasma": np.ascontiguousarray(f.variables["zetal_plasma"][()]).astype(
            float
        ),
        "zetal_coil": np.ascontiguousarray(f.variables["zetal_coil"][()]).astype(float),
        "r_plasma": np.ascontiguousarray(f.variables["r_plasma"][()]).astype(float),
        "r_coil": np.ascontiguousarray(f.variables["r_coil"][()]).astype(float),
        "chi2_B": np.ascontiguousarray(f.variables["chi2_B"][()]).astype(float),
        "single_valued_current_potential_thetazeta": np.ascontiguousarray(
            f.variables["single_valued_current_potential_thetazeta"][()]
        ).astype(float),
        "current_potential": np.ascontiguousarray(
            f.variables["current_potential"][()]
        ).astype(float),
        "Bnormal_from_plasma_current": np.ascontiguousarray(
            f.variables["Bnormal_from_plasma_current"][()]
        ),
        "Bnormal_from_net_coil_currents": np.ascontiguousarray(
            f.variables["Bnormal_from_net_coil_currents"][()]
        ).astype(float),
        "Bnormal_total": np.ascontiguousarray(f.variables["Bnormal_total"][()]).astype(
            float
        ),
        "net_poloidal_current_Amperes": np.ascontiguousarray(
            f.variables["net_poloidal_current_Amperes"][()]
        ).astype(float),
    }

    try:
        data["nlambda"] = int(f.variables["nlambda"][()])
        data["lambdas"] = np.ascontiguousarray(
            f.variables["lambda"][()]).astype(float)
    except KeyError:
        data["nlambda"] = int(f.variables["nalpha"][()])
        data["lambdas"] = np.ascontiguousarray(
            f.variables["alpha"][()]).astype(float)

    try:
        data["chi2_K"] = np.ascontiguousarray(
            f.variables["chi2_K"][()]).astype(float)
        data["K2"] = np.ascontiguousarray(f.variables["K2"][()]).astype(float)
    except KeyError:
        data["chi2_K"] = np.ascontiguousarray(
            f.variables["chi2_J"][()]).astype(float)
        data["K2"] = np.ascontiguousarray(f.variables["J2"][()]).astype(float)

    f.close()

    if np.max(np.abs(data["lambdas"])) < 1.0e-200:
        data["lambdas"] += 1

    permutation = np.argsort(data["lambdas"])
    data["lambdas"] = data["lambdas"][permutation]
    data["chi2_K"] = data["chi2_K"][permutation]
    data["chi2_B"] = data["chi2_B"][permutation]
    data["Bnormal_total"] = data["Bnormal_total"][permutation, :, :]
    data["single_valued_current_potential_thetazeta"] = data[
        "single_valued_current_potential_thetazeta"
    ][permutation, :, :]
    data["K2"] = data["K2"][permutation, :, :]
    data["current_potential"] = data["current_potential"][permutation, :, :]

    if data["lambdas"][-1] > 1.0e199:
        data["lambdas"][-1] = np.inf

    return data


def plot_current_contours(filepath, ilambda=0):
    data = load_netcdf_data(filepath)
    potentials_for_lambda = data["current_potential"]
    fig = go.Figure(go.Contour(z=potentials_for_lambda[ilambda]))
    return fig


def plot_current_contours_surface(filepath, ilambda=0, figure=None, num_coils_per_hp=0):
    data = load_netcdf_data(filepath)
    potentials_for_lambda = data["current_potential"][ilambda]
    if num_coils_per_hp > 0:
        bins = np.linspace(
            np.min(potentials_for_lambda),
            np.max(potentials_for_lambda),
            num_coils_per_hp * 2 + 2,
        )
        potentials_for_lambda = np.digitize(potentials_for_lambda, bins)
    potentials_for_lambda = np.tile(potentials_for_lambda, (data["nfp"], 1))
    r_array = data["r_coil"]

    fig = figure if isinstance(figure, go.Figure) else go.Figure()

    fig.add_trace(
        go.Surface(
            x=r_array[:, :, 0],
            y=r_array[:, :, 1],
            z=r_array[:, :, 2],
            surfacecolor=potentials_for_lambda,
        )
    )
    return fig


def get_cross_section(r_array, zetal_old, zeta_new, nfp):
    zetal_old = np.concatenate((zetal_old - nfp, zetal_old))
    r_array = np.concatenate((r_array, r_array), axis=0)

    x = r_array[:, :, 0]
    y = r_array[:, :, 1]
    z = r_array[:, :, 2]
    r = np.sqrt(x**2 + y**2)

    ntheta = z.shape[1]
    nzeta_new = len(zeta_new)
    r_slice = np.zeros([nzeta_new, ntheta])
    z_slice = np.zeros([nzeta_new, ntheta])
    for itheta in range(ntheta):
        interpolator = interp1d(zetal_old, r[:, itheta])
        r_slice[:, itheta] = interpolator(zeta_new)
        interpolator = interp1d(zetal_old, z[:, itheta])
        z_slice[:, itheta] = interpolator(zeta_new)

    return r_slice, z_slice


def plot_surfaces(filepath):
    data = load_netcdf_data(filepath)

    zeta_slices = np.array([0, 0.25, 0.5, 0.75]) * 2 * np.pi / data["nfp"]
    r_slice_plasma, z_slice_plasma = get_cross_section(
        data["r_plasma"], data["zetal_plasma"], zeta_slices, data["nfp"]
    )
    r_slice_coil, z_slice_coil = get_cross_section(
        data["r_coil"], data["zetal_coil"], zeta_slices, data["nfp"]
    )

    figs = []
    for which_plot in range(4):
        fig = go.Figure()
        zeta = zeta_slices[which_plot]
        fig.add_trace(
            go.Scatter(
                x=r_slice_coil[which_plot, :],
                y=z_slice_coil[which_plot, :],
                mode="lines+markers",
                name="coil",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=r_slice_plasma[which_plot, :],
                y=z_slice_plasma[which_plot, :],
                mode="lines+markers",
                name="plasma",
            )
        )
        fig.update_layout(
            title=f"zeta={zeta}",
            xaxis_title="R [meters]",
            yaxis_title="Z [meters]",
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
        )
        figs.append(fig)

    return figs


def plot_chi2(filepath):
    data = load_netcdf_data(filepath)
    nlambda = data["nlambda"]
    print(nlambda, type(nlambda))

    figs = []

    fig = px.scatter(
        x=data["chi2_K"],
        y=data["chi2_B"],
        log_x=True,
        log_y=True,
        labels={"x": "chi2_K [A^2]", "y": "chi2_B [T^2 m^2]"},
    )
    # for j in range(numPlots):
    fig.add_trace(
        go.Scatter(
            x=data["chi2_K"],
            y=data["chi2_B"],
            mode="markers",
            marker=dict(color="blue"),
        )
    )
    figs.append(fig)

    if nlambda > 1:
        fig = px.scatter(
            x=data["lambdas"],
            y=data["chi2_B"],
            log_x=True,
            log_y=True,
            labels={"x": "lambda [T^2 m^2 / A^2]", "y": "chi2_B [T^2 m^2]"},
        )
        fig.add_trace(
            go.Scatter(
                x=data["lambdas"],
                y=data["chi2_B"],
                mode="markers",
            )
        )
        figs.append(fig)

        fig = px.scatter(
            x=data["lambdas"],
            y=data["chi2_B"],
            log_x=False,
            log_y=True,
            labels={"x": "lambda [T^2 m^2 / A^2]", "y": "chi2_B [T^2 m^2]"},
        )
        fig.add_trace(
            go.Scatter(
                x=data["lambdas"],
                y=data["chi2_B"],
                mode="markers",
            )
        )
        figs.append(fig)

        fig = px.scatter(
            x=data["lambdas"],
            y=data["chi2_K"],
            log_x=True,
            log_y=True,
            labels={"x": "lambda [T^2 m^2 / A^2]", "y": "chi2_K [A^2]"},
        )
        fig.add_trace(
            go.Scatter(
                x=data["lambdas"],
                y=data["chi2_K"],
                mode="markers",
            )
        )
        figs.append(fig)

        fig = px.scatter(
            x=data["lambdas"],
            y=data["chi2_K"],
            log_x=False,
            log_y=True,
            labels={"x": "lambda [T^2 m^2 / A^2]", "y": "chi2_K [A^2]"},
        )
        fig.add_trace(
            go.Scatter(
                x=data["lambdas"],
                y=data["chi2_K"],
                mode="markers",
            )
        )
        figs.append(fig)

    return figs
