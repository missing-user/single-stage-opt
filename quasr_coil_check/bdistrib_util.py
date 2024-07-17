import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.io import netcdf_file
import pandas as pd
import plotly.graph_objects as go


def minimum_coil_surf_distance(curves, lcfs) -> float:
    min_dist = np.inf
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    for c in curves:
        pointcloud2 = c.gamma()
        min_dist = min(min_dist, np.min(cdist(pointcloud1, pointcloud2)))
        # Equivalent python code:
        # for point in pointcloud2:
        #   min_dist = min(min_dist, np.min(np.linalg.norm(pointcloud1 - point, axis=-1)))

    return float(min_dist)


def fit_exponential_rate(sequence):
    x = np.linspace(0, 1, len(sequence))
    fit = np.polyfit(x, np.log(sequence), 1)
    return fit


def rate_of_efficiency_sequence(
    bdistrib_path: str, plot=False, max_index_for_fit=150
) -> dict:
    """max_index_for_fit is useful, because the efficiency sequence gets corrupted by numerical noise for very large indices."""
    with netcdf_file(bdistrib_path, "r", mmap=False) as f:
        results = {}
        for B_key, svd_s_key in [
            (
                "Bnormal_from_1_over_R_field_inductance",
                "svd_s_inductance_plasma_middle",
            ),
            ("Bnormal_from_1_over_R_field_inductance", "svd_s_inductance_plasma_outer"),
            ("Bnormal_from_1_over_R_field_transfer", "svd_s_transferMatrix"),
            # "Bnormal_from_const_v_coils",
            # "Bnormal_from_const_v_coils_inductance",
            # "Bnormal_from_const_v_coils_transfer",
            # "Bnormal_from_plasma_current",
            # "Bnormal_from_plasma_current_inductance",
            # "Bnormal_from_plasma_current_transfer",
        ]:
            sequence = f.variables[B_key][()]
            svd_s = f.variables[svd_s_key][()].flatten()
            efficiency_seq = np.abs(sequence)
            feasibility_seq = efficiency_seq / svd_s
            eff_fit = fit_exponential_rate(efficiency_seq[:max_index_for_fit])
            feas_fit = fit_exponential_rate(feasibility_seq[:max_index_for_fit])

            eff_key = "efficiency " + svd_s_key.split("_")[-1]
            feas_key = "feasibility " + svd_s_key.split("_")[-1]

            if plot:
                results[eff_key] = efficiency_seq
                results[eff_key + " (fit)"] = np.exp(
                    np.polyval(eff_fit, x=np.linspace(0, 1, max_index_for_fit))
                )
                results[feas_key] = feasibility_seq
                results[feas_key + " (fit)"] = np.exp(
                    np.polyval(feas_fit, x=np.linspace(0, 1, max_index_for_fit))
                )
            else:
                results[eff_key] = eff_fit[0]
                results[feas_key] = feas_fit[0]
        return results


def plot_bdistrib_surfaces(bdistrib_output_path: str, figure=None) -> go.Figure:
    with netcdf_file(bdistrib_output_path, "r", mmap=False) as f:
        # Necessary casts to fix endianness issues between netcdf and numpy
        r_plasma = np.ascontiguousarray(f.variables["r_plasma"][()]).astype(float)
        r_middle = np.ascontiguousarray(f.variables["r_middle"][()]).astype(float)
        r_outer = np.ascontiguousarray(f.variables["r_outer"][()]).astype(float)

        def add_surface_trace(fig: go.Figure, r_data, fraction, name, **kwargs):
            fraction = 1.0 - fraction
            fig.add_trace(
                go.Surface(
                    x=r_data[int(r_data.shape[0] * fraction) :, :, 0],
                    y=r_data[int(r_data.shape[0] * fraction) :, :, 1],
                    z=r_data[int(r_data.shape[0] * fraction) :, :, 2],
                    name=name,
                    **kwargs
                )
            )

        fig = figure if isinstance(figure, go.Figure) else go.Figure()

        add_surface_trace(fig, r_plasma, 0.66, "r_plasma", colorscale="Plasma")
        add_surface_trace(fig, r_middle, 0.5, "r_middle", colorscale="Viridis")
        add_surface_trace(fig, r_outer, 0.33, "r_outer", colorscale="Plasma")
        return fig


def sanitize_df_for_analysis(simsopt_loaded_list: list) -> pd.DataFrame:
    df = pd.DataFrame(simsopt_loaded_list)
    df["lcfs"] = df["surfaces"].map(lambda x: x[-1])
    df["AR"] = df["lcfs"].map(lambda x: x.aspect_ratio())
    df["volume"] = df["lcfs"].map(lambda x: -x.volume())
    df["R1"] = df["lcfs"].map(lambda x: x.minor_radius())
    return df
