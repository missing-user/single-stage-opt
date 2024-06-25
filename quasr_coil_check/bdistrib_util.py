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


def rate_of_efficiency_sequence(
    bdistrib_path: str, plot=False, max_index_for_fit=45
) -> dict:
    """max_index_for_fit is useful, because the efficiency sequence gets corrupted by numerical noise for very large indices."""
    with netcdf_file(bdistrib_path, "r", mmap=False) as f:
        results = {}
        for variable_name in [
            "Bnormal_from_1_over_R_field",
            "Bnormal_from_1_over_R_field_inductance",
            "Bnormal_from_1_over_R_field_transfer",
            "Bnormal_from_const_v_coils",
            "Bnormal_from_const_v_coils_inductance",
            "Bnormal_from_const_v_coils_transfer",
            "Bnormal_from_plasma_current",
            "Bnormal_from_plasma_current_inductance",
            "Bnormal_from_plasma_current_transfer",
        ]:
            efficiency_sequence = f.variables[variable_name][()]
            efficiency_sequence = np.abs(efficiency_sequence)
            log_efficiency_sequence = np.log(efficiency_sequence)[:max_index_for_fit]
            x = np.linspace(
                0, len(log_efficiency_sequence), len(log_efficiency_sequence)
            )
            fit = np.polyfit(x, log_efficiency_sequence, 1)
            rate_of_increase = fit[0]
            if plot:
                results[variable_name] = efficiency_sequence
                results[variable_name + " (fit)"] = np.exp(np.polyval(fit, x))
            else:
                results[variable_name] = rate_of_increase
                # results[variable_name + " (dev)"] = np.std(
                #     log_efficiency_sequence - np.polyval(fit, x)
                # )
        return results


def plot_bdistrib_surfaces(bdistrib_path: str, figure=None) -> go.Figure:
    with netcdf_file(bdistrib_path, "r", mmap=False) as f:
        # Necessary casts to fix endianness issues between netcdf and numpy
        r_plasma = np.ascontiguousarray(f.variables["r_plasma"][()]).astype(float)
        r_middle = np.ascontiguousarray(f.variables["r_middle"][()]).astype(float)
        r_outer = np.ascontiguousarray(f.variables["r_outer"][()]).astype(float)

        def add_surface_trace(fig: go.Figure, r_data, name, **kwargs):
            fig.add_trace(
                go.Surface(
                    x=r_data[r_data.shape[0] // 2 :, :, 0],
                    y=r_data[r_data.shape[0] // 2 :, :, 1],
                    z=r_data[r_data.shape[0] // 2 :, :, 2],
                    name=name,
                    **kwargs
                )
            )

        fig = figure if isinstance(figure, go.Figure) else go.Figure()

        add_surface_trace(fig, r_plasma, "r_plasma", colorscale="Plasma")
        add_surface_trace(fig, r_middle, "r_middle", colorscale="Viridis")
        add_surface_trace(fig, r_outer, "r_outer", colorscale="Plasma")
        return fig


def sanitize_df_for_analysis(simsopt_loaded_list: list) -> pd.DataFrame:
    df = pd.DataFrame(simsopt_loaded_list)
    df["lcfs"] = df["surfaces"].map(lambda x: x[-1])
    df["AR"] = df["lcfs"].map(lambda x: x.aspect_ratio())
    df["volume"] = df["lcfs"].map(lambda x: -x.volume())
    df["nfp"] = df["lcfs"].map(lambda x: int(x.nfp))
    df["R1"] = df["lcfs"].map(lambda x: x.minor_radius())
    return df
