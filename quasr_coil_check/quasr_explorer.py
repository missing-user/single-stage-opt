import os
import numpy as np
import pandas as pd
import simsopt
import simsopt.geo

import dash
import diskcache
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import precompute_complexities
import precompute_regcoil
import regcoil_plot
import bdistrib_util
import bdistrib_io

pio.templates.default = "plotly_dark"


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
                        dash.dcc.Input(
                            1150000,
                            type="number",
                            id="num-dataset-input",
                            persistence=True,
                        ),
                        dash.dcc.Dropdown(
                            id="x-axis-select",
                            persistence=True,
                        ),
                        dash.dcc.Dropdown(
                            id="y-axis-select",
                            multi=True,
                            persistence=True,
                        ),
                        dash.dcc.Checklist(
                            ["xlog", "ylog"],
                            [],
                            inline=True,
                            id="log-axis-select",
                            persistence=True,
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dash.dcc.Dropdown(
                            id="bdistrib-matrix-dropdown",
                            options=["inductance", "transfer"],
                            value="inductance",
                            persistence=True,
                        ),
                        dash.dcc.Dropdown(
                            id="sequence-dropdown",
                            options=["efficiency", "feasibility"],
                            value="efficiency",
                            persistence=True,
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(dash.html.Progress(id="progress-bar")),
        dbc.Row(
            [
                dbc.Col(
                    dash.dcc.Loading(
                        dash.dcc.Graph(id="metric-scatter-plot", figure={})
                    )
                ),
                dbc.Col(
                    dash.dcc.Loading(dash.dcc.Graph(
                        id="correlation-plot", figure={}))
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dash.dcc.Loading(
                        dash.dcc.Graph(
                            id="efficiency-sequence-plot", figure={})
                    )
                ),
                dbc.Col(
                    [
                        dash.dcc.Dropdown(
                            id="plasma-surface-dropdown",
                            options=["lcfs", "bdistrib surfaces",
                                     "regcoil surfaces"],
                            value="lcfs",
                        ),
                        dash.dcc.Loading(
                            dash.dcc.Graph(id="plasma-surface-plot", figure={})
                        ),
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
    dash.Input("num-dataset-input", "value"),
    running=[
        (
            dash.Output("progress-bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
    ],
    manager=long_callback_manager,
    progress=[dash.Output("progress-bar", "value"),
              dash.Output("progress-bar", "max")],
)
def load_results(set_progress, max_ID):
    max_ID = int(max_ID)

    set_progress(("1", str(2 * max_ID)))
    # simsopt_objs = bdistrib_io.load_simsopt_up_to(max_ID)
    set_progress(("2", "5"))
    df: pd.DataFrame = pd.read_pickle("QUASR_db/QA_database_26032024.pkl")
    df["complexity"] = df["max_kappa"] + df["max_msc"]
    df["log(qs error)"] = np.log(df["qs_error"])
    print("Raw dataset has", len(df), "entries")

    # df = bdistrib_util.sanitize_df_for_analysis(simsopt_objs)
    set_progress(("3", "5"))

    efficiencies = []
    for ID in range(max_ID):
        bdistrib_path = bdistrib_io.get_file_path(ID, "bdistrib")
        if os.path.exists(bdistrib_path):
            results_dict = bdistrib_util.rate_of_efficiency_sequence(
                bdistrib_path)
            results_dict["ID"] = ID
            efficiencies.append(results_dict)
            set_progress((str(ID), str(2 * max_ID)))
    # df = df.merge(pd.DataFrame(efficiencies), left_on="ID", right_on="ID")
    print("Loaded", len(efficiencies), "efficiency sequences")

    # Compute coil complexity from coils
    def get_complexity(ID):
        set_progress((str(max_ID + ID), str(2 * max_ID)))
        compl = precompute_complexities.cached_get_complexity(ID)
        compl.pop("nfp", None)
        return compl

    # complexity = [get_complexity(ID) for ID in df["ID"]]
    # df = df.merge(pd.DataFrame(complexity), left_on="ID", right_on="ID")
    # print("Loaded", len(complexity ), "complexities for analysis")

    regcoils = []
    for ID in df["ID"]:
        regcoil_path = bdistrib_io.get_file_path(ID, "regcoil")
        if os.path.exists(regcoil_path):
            results_dict = precompute_regcoil.get_regcoil_metrics(ID)
            regcoils.append(results_dict)
    df = df.merge(pd.DataFrame(regcoils), left_on="ID",
                  right_on="ID").infer_objects()
    print("Loaded", len(regcoils), "regcoils for analysis")
    print("Finally", len(df), "datasets for analysis")

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


@app.callback(dash.Output("correlation-plot", "figure"), dash.Input("df-store", "data"))
def correlationplot(dfstore):
    df = pd.DataFrame(dfstore).convert_dtypes()
    return px.imshow(df.corr(), height=600)


@app.callback(
    dash.Output("x-axis-select", "value"),
    dash.Output("y-axis-select", "value"),
    dash.Input("correlation-plot", "clickData"),
)
def select_correlation(clickdata):
    return clickdata["points"][0]["x"], [p["y"] for p in clickdata["points"]]


@app.callback(
    dash.Output("metric-scatter-plot", "figure"),
    dash.Input("df-store", "data"),
    dash.Input("x-axis-select", "value"),
    dash.Input("y-axis-select", "value"),
    dash.Input("log-axis-select", "value"),
)
def scatterplot(dfstore, xscalar, yscalar, logatirhmic):
    df = pd.DataFrame(dfstore).convert_dtypes()
    df["nfp"] = df["nfp"].astype(str)
    if len(yscalar) == 1:
        yscalar = yscalar[0]
        fig = px.scatter(
            df,
            xscalar,
            yscalar,
            color="nfp",
            log_x="xlog" in logatirhmic,
            log_y="ylog" in logatirhmic,
            hover_data={"ID": True},
            custom_data=["ID"],
        )
    else:
        fig = px.scatter_matrix(
            data_frame=df,
            dimensions=yscalar,
            color="nfp",
            hover_data={"ID": True},
        ).update_traces(diagonal_visible=True, showupperhalf=False)
    return fig.update_layout(clickmode="event", height=600)


@app.callback(
    dash.Output("efficiency-sequence-plot", "figure"),
    dash.Input("metric-scatter-plot", "hoverData"),
    dash.Input("bdistrib-matrix-dropdown", "value"),
    dash.Input("sequence-dropdown", "value"),
)
def efficiency_sequence_plot(hover_data, selected_matrix, selected_sequence):
    if not hover_data or not "customdata" in hover_data["points"][0]:
        return {}

    ID = int(hover_data["points"][0]["customdata"][0])

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
    dash.Output("plasma-surface-plot", "figure"),
    dash.Input("metric-scatter-plot", "clickData"),
    dash.Input("plasma-surface-dropdown", "value"),
)
def display_hover_data(hover_data, selected_surface_type):
    if not hover_data or not "customdata" in hover_data["points"][0]:
        return {}

    result_id = int(hover_data["points"][0]["customdata"][0])
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
    elif selected_surface_type == "regcoil surfaces":
        fig = regcoil_plot.plot_current_contours(
            bdistrib_io.get_file_path(result_id, "regcoil"), -1
        )

    # type: ignore
    return fig.update_layout(height=600, title=f"ID: {result_id}", scene=dict(aspectmode="data"))


app.run(debug=True, port="8051")
