import dash
import simsopt
import simsopt.geo
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import os
import precompute_complexities

import bdistrib_util
import bdistrib_io

pio.templates.default = "plotly_dark"

import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = dash.long_callback.DiskcacheLongCallbackManager(cache)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    long_callback_manager=long_callback_manager,
)

app.layout = dash.html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dash.dcc.Slider(0, 1150000, value=1092000, id="max_ID"),
                        dash.dcc.Dropdown(id="scalarselect"),
                    ]
                ),
                dbc.Col(
                    [
                        dash.dcc.Dropdown(
                            id="surface_select",
                            options=["lcfs", "bdistrib surfaces"],
                            value="lcfs",
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(dash.html.Progress(id="progress_bar")),
        dbc.Row(
            [
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="figure1", figure={}))),
                dbc.Col(
                    dash.dcc.Loading(
                        dash.dcc.Graph(id="efficiency_sequence", figure={})
                    )
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="hover_fig", figure={}))),
            ]
        ),
        dash.dcc.Store(id="df-store"),
    ],
    **{"data-bs-theme": "dark"},
)


@app.long_callback(
    dash.Output("df-store", "data"),
    dash.Input("max_ID", "value"),
    running=[
        (
            dash.Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
    ],
    manager=long_callback_manager,
    progress=[dash.Output("progress_bar", "value"), dash.Output("progress_bar", "max")],
)
def load_results(set_progress, max_ID):
    max_ID = int(max_ID)

    set_progress(("1", str(2 * max_ID)))
    # simsopt_objs = bdistrib_io.load_simsopt_up_to(max_ID)
    set_progress(("2", "5"))
    # df = bdistrib_util.sanitize_df_for_analysis(simsopt_objs)
    set_progress(("3", "5"))

    efficiencies = []
    for ID in range(max_ID):
        bdistrib_path = bdistrib_io.get_file_path(ID, "bdistrib")
        if os.path.exists(bdistrib_path):
            results_dict = bdistrib_util.rate_of_efficiency_sequence(bdistrib_path)
            results_dict["ID"] = ID
            efficiencies.append(results_dict)
            set_progress((str(ID), str(2 * max_ID)))
    df = pd.DataFrame(efficiencies)

    # Compute coil complexity from coils
    def get_complexity(ID):
        set_progress((str(max_ID + ID), str(2 * max_ID)))
        return precompute_complexities.cached_get_complexity(ID)

    complexity = [get_complexity(ID) for ID in df["ID"]]
    df = df.join(pd.DataFrame(complexity)).select_dtypes(exclude=["object"])
    return df.to_dict("records")


@app.callback(dash.Output("scalarselect", "options"), dash.Input("df-store", "data"))
def dropdown(dfstore):
    df = pd.DataFrame(dfstore).convert_dtypes()
    return df.select_dtypes(include=["number"]).columns.tolist()


@app.callback(
    dash.Output("figure1", "figure"),
    dash.Input("df-store", "data"),
    dash.Input("scalarselect", "value"),
)
def scatterplot(dfstore, xscalar):
    df = pd.DataFrame(dfstore).convert_dtypes()
    df["nfp"] = df["nfp"].astype(str)
    fig = px.scatter(
        df,
        xscalar,
        "complexity",
        color="nfp",
        hover_data={"ID": True},
    )
    return fig.update_layout(clickmode="event", height=600)


@app.callback(
    dash.Output("efficiency_sequence", "figure"),
    dash.Input("figure1", "hoverData"),
    dash.Input("scalarselect", "value"),
)
def display_hover_data1(hoverData, selected_scalar):
    if not hoverData:
        return {}

    ID = int(hoverData["points"][0]["customdata"][0])
    results = bdistrib_util.rate_of_efficiency_sequence(
        bdistrib_io.get_file_path(ID, "bdistrib"), plot=True
    )

    fig = go.Figure(
        [
            go.Scatter(
                y=results[selected_scalar], mode="lines", name="Efficiency Sequence"
            ),
            go.Scatter(
                y=results[selected_scalar + " (fit)"], mode="lines", name="Fit Line"
            ),
        ]
    )
    fig.update_layout(
        # title=str(rate_of_increase),
        yaxis_type="log",
        xaxis_title="Index",
        yaxis_title="Value",
    )

    return fig


@app.callback(
    dash.Output("hover_fig", "figure"),
    dash.Input("figure1", "clickData"),
    dash.Input("surface_select", "value"),
)
def display_hover_data(hoverData, selected_surface_type):
    if not hoverData:
        return {}

    result_id = int(hoverData["points"][0]["customdata"][0])
    optimization_res = simsopt.load(bdistrib_io.get_file_path(result_id))

    plotting_surf = [optimization_res[0][-1]] if selected_surface_type == "lcfs" else []
    fig = simsopt.geo.plot(
        plotting_surf + optimization_res[1],
        show=False,
        engine="plotly",
        close=True,
    )
    if selected_surface_type == "bdistrib surfaces":
        fig = bdistrib_util.plot_bdistrib_surfaces(
            bdistrib_io.get_file_path(result_id, "bdistrib"), figure=fig
        )

    return fig.update_layout(height=600, title=f"ID: {result_id}")  # type: ignore


app.run(debug=True)
