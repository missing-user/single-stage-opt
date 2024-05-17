import dash
import simsopt
import simsopt.geo
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json

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
            dash.dcc.Dropdown(options=all_paths, value=all_paths[0], id="fileselect")
        ),
        dbc.Row(dash.html.Progress(id="progress_bar")),
        dbc.Row(
            [
                dbc.Col(
                    dash.dcc.Loading(
                        dash.dcc.Graph(
                            id="figure1",
                        )
                    )
                ),
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="hover_fig"))),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="BdotN"))),
                dbc.Col(dash.dcc.Loading(dash.dcc.Graph(id="Btarget"))),
            ]
        ),
    ]
)


@app.long_callback(
    dash.Output("figure1", "figure"),
    dash.Input("fileselect", "value"),
    running=[
        (
            dash.Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
    ],
    progress=[dash.Output("progress_bar", "value"), dash.Output("progress_bar", "max")],
    prevent_intial_callback=True,
)
def load_results(set_progress, filepath):
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
    df["success"] = df["B.n residual max"] < df["B target max"] / 2
    fig = px.scatter(
        df.select_dtypes(exclude=["object"]),
        "spectral_radius",
        "complexity",
        color="B.n residual max",
        range_color=[0, 2],
        symbol="success",
        hover_data={"filename": True},
        template="plotly_dark",
    )
    return fig.update_layout(height=600, clickmode="event+select")


@app.callback(
    dash.Output("BdotN", "figure"),
    dash.Output("Btarget", "figure"),
    dash.Input("figure1", "hoverData"),
    dash.Input("fileselect", "value"),
)
def display_hover_data1(hoverData, filepath):
    if not hoverData or not filepath:
        return None, None
    print(hoverData, filepath)

    result_path = str(hoverData["points"][0]["customdata"][0])
    optimization_res = simsopt.load(os.path.join(filepath, result_path))
    return (
        px.imshow(
            optimization_res["BdotN"],
            title="B.n residual",
            # template="plotly_dark",
        ),
        px.imshow(
            optimization_res["B_external_normal"],
            title="B.n target",
            # template="plotly_dark",
        ),
    )


# @app.callback(
#     dash.Output("hover_fig", "figure"),
#     dash.Input("figure1", "clickData"),
#     dash.State("fileselect", "value"),
# )
# def display_hover_data(hoverData, filepath):
#     if not hoverData or not filepath:
#         return None

#     result_path = str(hoverData["points"][0]["customdata"][0])
#     optimization_res = simsopt.load(os.path.join(filepath, result_path))
#     # print(optimization_res)
#     # import plotly.graph_objects as go

#     # fig = go.Figure()
#     fig = simsopt.geo.plot(
#         [optimization_res["surf"]] + optimization_res["coils"],
#         # fig,
#         show=False,
#         engine="plotly",
#     )
#     print(fig)
#     return fig


app.run(debug=True)
