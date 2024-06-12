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


def rate_of_efficiency_sequence(bdistrib_path: str, plot=False) -> tuple:
    with netcdf_file(bdistrib_path, "r", mmap=False) as f:
        efficiency_sequence = f.variables["Bnormal_from_const_v_coils_inductance"][()]
        efficiency_sequence = np.abs(efficiency_sequence)
        log_efficiency_sequence = np.log(efficiency_sequence)
        x = np.linspace(0, len(log_efficiency_sequence), len(log_efficiency_sequence))
        fit = np.polyfit(x, log_efficiency_sequence, 1)
        rate_of_increase = fit[0]
        if plot:
            # plt.semilogy(efficiency_sequence)
            # plt.semilogy(np.exp(np.polyval(fit, x)))
            # plt.title(str(rate_of_increase))
            # plt.show()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=efficiency_sequence, mode="lines", name="Efficiency Sequence"
                )
            )
            fit_line = np.exp(np.polyval(fit, x))
            fig.add_trace(go.Scatter(y=fit_line, mode="lines", name="Fit Line"))
            fig.update_layout(
                title=str(rate_of_increase),
                yaxis_type="log",
                xaxis_title="Index",
                yaxis_title="Value",
            )
            return fig
        return rate_of_increase, np.std(log_efficiency_sequence - np.polyval(fit, x))


def sanitize_df_for_analysis(simsopt_loaded_list: list) -> pd.DataFrame:
    df = pd.DataFrame(simsopt_loaded_list)
    df["lcfs"] = df["surfaces"].map(lambda x: x[-1])
    df["AR"] = df["lcfs"].map(lambda x: x.aspect_ratio())
    df["volume"] = df["lcfs"].map(lambda x: -x.volume())
    df["nfp"] = df["lcfs"].map(lambda x: int(x.nfp))
    df["R1"] = df["lcfs"].map(lambda x: x.minor_radius())
    return df
