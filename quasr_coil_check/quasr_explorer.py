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
                        dash.dcc.Slider(0, 10000, value=1000, id="max_ID"),
                        dash.dcc.Dropdown(id="scalarselect"),
                    ]
                ),
                dbc.Col(
                    [
                        # dash.html.Label("Success Threshold"),
                        # dash.dcc.Slider(0, 1, 0.05, value=0.5, id="threshold-slider"),
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

    set_progress(("0", "4"))
    simsopt_objs = bdistrib_io.load_simsopt_up_to(max_ID)
    df = bdistrib_util.sanitize_df_for_analysis(simsopt_objs)
    set_progress(("1", "4"))

    efficiencies = []
    for ID in range(max_ID):
        bdistrib_path = bdistrib_io.get_file_path(ID, "bdistrib")
        if os.path.exists(bdistrib_path):
            avg, stddev = bdistrib_util.rate_of_efficiency_sequence(bdistrib_path)
            efficiencies.append(
                {
                    "efficiency sequence rate of increase": avg,
                    "efficiency sequence rate of increase (dev)": stddev,
                }
            )
    df = df.join(pd.DataFrame(efficiencies))
    set_progress(("2", "4"))

    # Compute coil complexity from coils
    complexity = []
    for idx, row in df.iterrows():
        s = row["lcfs"]
        curves = [c.curve for c in row["coils"]]

        # Form the total objective function.
        LENGTH_WEIGHT = 0.05
        TARGET_LENGTH = 1.0 * s.minor_radius() * 2 * np.pi
        CC_WEIGHT = 1.0
        CC_THRESHOLD = 0
        CURVATURE_THRESHOLD = 0
        CURVATURE_WEIGHT = 1e-3

        MSC_THRESHOLD = 0
        MSC_WEIGHT = 1e-3
        from simsopt.objectives import QuadraticPenalty, SquaredFlux

        Jls = LENGTH_WEIGHT * sum(
            [
                QuadraticPenalty(simsopt.geo.CurveLength(c), TARGET_LENGTH)
                for c in curves
            ]
        )
        Jccdist = CC_WEIGHT * simsopt.geo.CurveCurveDistance(curves, CC_THRESHOLD)
        Jcsdist = simsopt.geo.CurveSurfaceDistance(curves, s, s.minor_radius())
        Jcs = CURVATURE_WEIGHT * sum(
            [simsopt.geo.LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in curves]
        )
        Jmscs = MSC_WEIGHT * sum(
            [
                QuadraticPenalty(
                    simsopt.geo.MeanSquaredCurvature(c), MSC_THRESHOLD, "max"
                )
                for c in curves
            ]
        )

        JF = Jccdist + Jcsdist + Jcs + Jmscs  # +Jls +
        complexity.append(
            {
                "complexity": float(JF.J()),
                "Jls": Jls.J(),
                "Jccdist": Jccdist.J(),
                "Jcs": Jcs.J(),
                "Jmscs": Jmscs.J(),
            }
        )

    set_progress(("3", "4"))
    df = df.join(pd.DataFrame(complexity))
    print(df.head())

    return df.select_dtypes(exclude=["object"]).to_dict("records")


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
)
def display_hover_data1(hoverData):
    if not hoverData:
        return {}

    ID = int(hoverData["points"][0]["customdata"][0])
    return bdistrib_util.rate_of_efficiency_sequence(
        bdistrib_io.get_file_path(ID, "bdistrib"), plot=True
    )


@app.callback(
    dash.Output("hover_fig", "figure"),
    dash.Input("figure1", "clickData"),
)
def display_hover_data(hoverData):
    if not hoverData:
        return {}

    result_id = int(hoverData["points"][0]["customdata"][0])
    optimization_res = simsopt.load(bdistrib_io.get_file_path(result_id))

    return simsopt.geo.plot(
        optimization_res[0] + optimization_res[1],
        show=False,
        engine="plotly",
        close=True,
    ).update_layout(height=600, title=f"ID: {result_id}")


app.run(debug=True)
