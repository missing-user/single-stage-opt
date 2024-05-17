import dash
import simsopt
import simsopt.geo
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

import plotly.io as pio
import plotly.express as px
import os
import json

pio.templates.default = "plotly_dark"

import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = dash.long_callback.DiskcacheLongCallbackManager(cache)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    long_callback_manager=long_callback_manager,
)

out_dir = "output"
directories = [
    os.path.join(out_dir, name)
    for name in os.listdir(out_dir)
    if os.path.isdir(os.path.join(out_dir, name))
]
all_paths = list(reversed(sorted(directories)))

app.layout = dash.html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dash.dcc.Dropdown(
                        options=all_paths, value=all_paths[0], id="fileselect"
                    )
                ),
                dbc.Col(dash.dcc.Slider(0, 1, 0.05, value=0.5, id="threshold-slider")),
            ]
        ),
        dbc.Row(dash.html.Progress(id="progress_bar")),
        dbc.Row(
            [
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="figure1", figure={}))),
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="hover_fig", figure={}))),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="BdotN", figure={}))),
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="Btarget", figure={}))),
                # dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="BdotNfft", figure={}))),
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="Btargetfft", figure={}))),
            ]
        ),
    ],
    **{"data-bs-theme": "dark"}
)


@app.long_callback(
    dash.Output("figure1", "figure"),
    dash.Input("fileselect", "value"),
    dash.Input("threshold-slider", "value"),
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
def load_results(set_progress, filepath, success_threshold):
    analysis_folder = os.path.join(filepath)
    results = []
    i = 0
    all_files = os.listdir(analysis_folder)
    for i, path in enumerate(all_files):
        if path.endswith(".json"):
            optimization_res = simsopt.load(analysis_folder + "/" + path)
            optimization_res["filename"] = str(path)
            results.append(optimization_res)

        set_progress((str(i + 1), str(len(all_files))))

    df = pd.DataFrame(results).convert_dtypes()
    df["J"] = df["J"].astype(float)
    df["B target max"] = df["B_external_normal"].apply(np.max)
    df["B.n residual max"] = df["BdotN"].apply(np.max)
    df["spectral_radius"] = df["spectral_radius"].astype(float)
    df["complexity"] = df["complexity"].astype(float)
    df["magnitude"] = df["magnitude"].astype(float)
    # df["run_id"] = df.index
    df["success"] = df["B.n residual max"] < df["B target max"] * success_threshold
    fig = px.scatter(
        df.select_dtypes(exclude=["object"]),
        "spectral_radius",
        "complexity",
        color="B.n residual max",
        range_color=[0, 2],
        symbol="success",
        # symbol_sequence=["circle", "x"],
        hover_data={"filename": True},
    )
    return fig.update_layout(clickmode="event", height=600)


@app.callback(
    dash.Output("BdotN", "figure"),
    dash.Output("Btarget", "figure"),
    # dash.Output("BdotNfft", "figure"),
    dash.Output("Btargetfft", "figure"),
    dash.Input("figure1", "hoverData"),
    dash.Input("fileselect", "value"),
)
def display_hover_data1(hoverData, filepath):
    if not hoverData or not filepath:
        return {}, {}, {}

    result_path = str(hoverData["points"][0]["customdata"][0])
    optimization_res = simsopt.load(os.path.join(filepath, result_path))

    targetmin, targetmax = np.min(optimization_res["B_external_normal"]), np.max(
        optimization_res["B_external_normal"]
    )
    return (
        px.imshow(
            optimization_res["BdotN"],
            title="B.n residual",
            range_color=[targetmin, targetmax],
        ),
        px.imshow(
            optimization_res["B_external_normal"],
            title="B.n target",
        ),
        px.imshow(
            np.fft.fftshift(np.fft.fft2(optimization_res["B_external_normal"])).imag,
            title="B.n target FFT",
        ),
    )


@app.callback(
    dash.Output("hover_fig", "figure"),
    dash.Input("figure1", "clickData"),
    dash.State("fileselect", "value"),
)
def display_hover_data(hoverData, filepath):
    if not hoverData or not filepath:
        return {}

    result_path = str(hoverData["points"][0]["customdata"][0])
    optimization_res = simsopt.load(os.path.join(filepath, result_path))

    return simsopt.geo.plot(
        [optimization_res["surf"]] + optimization_res["coils"],
        show=False,
        engine="plotly",
    ).update_layout(height=600)


app.run(debug=True)
