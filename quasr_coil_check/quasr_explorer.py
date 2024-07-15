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
                        dash.dcc.Dropdown(id="x-axis-select"),
                        dash.dcc.Dropdown(id="y-axis-select", value="complexity"),
                    ]
                ),
                dbc.Col(
                    [
                        dash.dcc.Dropdown(
                            id="matrix_select",
                            options=["inductance", "transfer"],
                            value="inductance",
                        ),
                        dash.dcc.Dropdown(
                            id="sequence_select",
                            options=["efficiency", "feasibility"],
                            value="efficiency",
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
                dbc.Col(
                    [
                        dash.dcc.Dropdown(
                            id="surface_select",
                            options=["lcfs", "bdistrib surfaces"],
                            value="lcfs",
                        ),
                        dash.dcc.Loading(dash.dcc.Graph(id="hover_fig", figure={})),
                    ]
                ),
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
    unpickled_df = pd.read_pickle("QUASR_db/QA_database_26032024.pkl")

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

    # complexity = [get_complexity(ID) for ID in df["ID"]]
    # df = df.join(pd.DataFrame(complexity)).select_dtypes(exclude=["object"])
    unpickled_df["complexity"] = unpickled_df["max_kappa"] + unpickled_df["max_msc"]
    unpickled_df["log(qs error)"] = np.log(unpickled_df["qs_error"])

    df = df.merge(unpickled_df, left_on="ID", right_on="ID")

    return df.select_dtypes(exclude=["object"]).to_dict("records")


@app.callback(
    dash.Output("x-axis-select", "options"),
    dash.Output("y-axis-select", "options"),
    dash.Input("df-store", "data"),
)
def dropdown(dfstore):
    df = pd.DataFrame(dfstore).convert_dtypes()
    axis_options = df.select_dtypes(include=["number"]).columns.tolist()
    return axis_options, axis_options


@app.callback(
    dash.Output("figure1", "figure"),
    dash.Input("df-store", "data"),
    dash.Input("x-axis-select", "value"),
    dash.Input("y-axis-select", "value"),
)
def scatterplot(dfstore, xscalar, yscalar):
    df = pd.DataFrame(dfstore).convert_dtypes()
    df["nfp"] = df["nfp"].astype(str)
    fig = px.scatter(
        df,
        xscalar,
        yscalar,
        color="nfp",
        hover_data={"ID": True},
        custom_data=["ID"],
    )
    return fig.update_layout(clickmode="event", height=600)


@app.callback(
    dash.Output("efficiency_sequence", "figure"),
    dash.Input("figure1", "hoverData"),
    dash.Input("matrix_select", "value"),
    dash.Input("sequence_select", "value"),
)
def display_hover_data1(hoverData, selected_matrix, selected_sequence):
    if not hoverData or not "customdata" in hoverData["points"][0]:
        return {}

    ID = int(hoverData["points"][0]["customdata"][0])

    results = bdistrib_util.rate_of_efficiency_sequence(
        bdistrib_io.get_file_path(ID, "bdistrib"), plot=True
    )
    # Convert the results dictionary into a DataFrame
    data = []
    for key, values in results.items():
        for i, value in enumerate(values):
            data.append(
                {
                    "Index": i,
                    "Value": value,
                    "Trace": key.replace(" (fit)", ""),
                    "Type": "Fit" if "(fit)" in key else "Raw",
                }
            )

    df = pd.DataFrame(data)

    # Filter the DataFrame to include only rows where the selected_sequence is in the Trace
    filtered_df = df[df["Trace"].str.contains(selected_sequence)]

    # Create the plot using Plotly Express
    fig = px.line(
        filtered_df,
        x="Index",
        y="Value",
        log_y=True,
        color="Trace",
        line_dash="Type",
        height=600,
    )
    return fig


@app.callback(
    dash.Output("hover_fig", "figure"),
    dash.Input("figure1", "clickData"),
    dash.Input("surface_select", "value"),
)
def display_hover_data(hoverData, selected_surface_type):
    if not hoverData or not "customdata" in hoverData["points"][0]:
        return {}

    result_id = int(hoverData["points"][0]["customdata"][0])
    optimization_res = simsopt.load(bdistrib_io.get_file_path(result_id))

    fig = None
    if selected_surface_type == "lcfs":
        fig = simsopt.geo.plot(
            [optimization_res[0][-1]] + optimization_res[1],
            show=False,
            engine="plotly",
            close=True,
        )
    elif selected_surface_type == "bdistrib surfaces":
        fig = bdistrib_util.plot_bdistrib_surfaces(
            bdistrib_io.get_file_path(result_id, "bdistrib"), figure=fig
        )

    return fig.update_layout(height=600, title=f"ID: {result_id}", scene=dict(aspectmode="data"))  # type: ignore


app.run(debug=True, port="8051")
