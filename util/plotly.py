import os
from os.path import join
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from util.trading import load_trades_from_pickle

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
trade_data_directory = join(cwd, "data", "03-1_trades") + os.sep

trade_data_directory
day = "2020-03-18"
tau_day = 9
trade_type = "K2"
x = 0.5


def translate_color(c):
    if c == "red":
        return "rgb(255,0,0)"
    elif c == "blue":
        return "rgb(0,0,255)"


def plotly_plot(trade_data_directory, day, tau_day, trade_type, x=0.5):
    data = load_trades_from_pickle(
        trade_data_directory, day, tau_day, trade_type
    )
    hd = data["hd"]
    rnd = data["rnd"]
    rnd_points = data["rnd_points"]
    kernel = data["kernel"]
    K_bound = data["K_bound"]
    M = data["M"]
    M_bounds_buy = data["M_bounds_buy"]
    M_bounds_sell = data["M_bounds_sell"]
    df_all = data["df_all"]
    df_trades = data["df_trades"]

    # Build figure
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Densities", "Kernel: rnd / hd")
    )

    colors = rnd_points.color.apply(lambda c: translate_color(c)).tolist()
    fig.add_trace(
        go.Scatter(
            x=rnd_points.M,
            y=rnd_points.q_M,
            mode="markers",
            name="options",
            opacity=0.4,
            marker=dict(
                size=7,
                symbol="circle",
                color=colors,
            ),
        ),
        row=1,
        col=1,
    )
    # Add scatter trace with medium sized markers
    fig.add_trace(
        go.Scatter(
            x=M,
            y=rnd,
            mode="lines",
            line=dict(color="rgb(255,0,0)"),
            name="rnd",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=M, y=hd, mode="lines", line=dict(color="rgb(0,0,255)"), name="hd"
        ),
        row=1,
        col=1,
    )

    # Kernel Plot
    fig.add_trace(  # K_bound
        go.Scatter(
            x=[1 - x, 1 - x, 1 + x, 1 + x],
            y=[1 - K_bound, 1 + K_bound, 1 + K_bound, 1 - K_bound],
            fill="toself",
            mode="lines",
            fillcolor="rgba(192,192,192,0.2)",
            line=dict(color="#0061ff", width=0),
        ),
        row=1,
        col=2,
    )
    for interval in M_bounds_buy:
        left = interval[0]
        right = interval[1]
        fig.add_trace(  # buy_areas
            go.Scatter(
                x=[left, left, right, right],
                y=[0, 2, 2, 0],
                fill="toself",
                mode="lines",
                fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="#0061ff", width=0),
            ),
            row=1,
            col=2,
        )

    for interval in M_bounds_sell:
        left = interval[0]
        right = interval[1]
        fig.add_trace(  # sell_areas
            go.Scatter(
                x=[left, left, right, right],
                y=[0, 2, 2, 0],
                fill="toself",
                mode="lines",
                fillcolor="rgba(0,0,255,0.1)",
                line=dict(color="#0061ff", width=0),
            ),
            row=1,
            col=2,
        )

    fig.add_trace(  # kernel
        go.Scatter(
            x=M,
            y=kernel,
            mode="lines",
            line=dict(color="black"),
            name="kernel",
        ),
        row=1,
        col=2,
    )

    colors = df_all.color.apply(lambda c: translate_color(c)).tolist()
    fig.add_trace(  # available options
        go.Scatter(
            x=df_all.M,
            y=[1] * len(df_all.M),
            mode="markers",
            name="options",
            opacity=0.4,
            marker=dict(
                size=7,
                symbol="circle",
                color=colors,
            ),
        ),
        row=1,
        col=2,
    )

    if df_trades is not None:
        colors = df_trades.color.apply(lambda c: translate_color(c)).tolist()
        fig.add_trace(  # trades
            go.Scatter(
                x=df_trades.M,
                y=[1] * len(df_trades.M),
                mode="markers",
                name="options",
                opacity=1,
                marker=dict(
                    size=10,
                    symbol="circle",
                    color=colors,
                ),
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(rangemode="nonnegative")
    fig.update_xaxes(range=[1 - x, 1 + x])
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        title_text="Moneyness", range=[1 - x, 1 + x], row=1, col=1
    )
    fig.update_xaxes(
        title_text="Moneyness", range=[1 - x, 1 + x], row=1, col=2
    )

    fig.update_yaxes(title_text=" ", rangemode="nonnegative", row=1, col=1)
    fig.update_yaxes(title_text=" ", range=[0, 2], row=1, col=2)
    fig.update_layout(
        template="none",
        height=400,
        width=1000,
        title_text="{}    {}    {}".format(day, tau_day, trade_type),
    )
    return fig


# fig = plotly_plot(trade_data_directory, day, tau_day, trade_type)
# fig.show()
