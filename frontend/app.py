import os
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Output, Input, dash_table
import plotly.express as px
import pandas as pd

# Get the date from the environment variable
date = os.getenv("DATE")
if not date:
    raise ValueError("The DATE environment variable is not set. Please define it before launching the app.")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix="/dash/",
    routes_pathname_prefix="/dash/"
)

colors = {'background': '#111111', 'text': '#7FDBFF'}

def get_participant_paths():
    """
    Returns a list of dicts with:
      - participant: name of the subdirectory
      - f1: path to the F1-Score metrics.csv (in test/<participant>)
      - cpu: path to the CPU percent metrics.csv (in metrics/<participant>)
      - bytes: path to the Bytes sent metrics.csv (in metrics/<participant>)
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_base = os.path.join(project_root, "robust", "logs", date, "test")
    metrics_base = os.path.join(project_root, "robust", "logs", date, "metrics")

    participants = set()
    if os.path.exists(test_base):
        participants |= set(os.listdir(test_base))
    if os.path.exists(metrics_base):
        participants |= set(os.listdir(metrics_base))

    items = []
    for participant in sorted(participants):
        test_csv = os.path.join(test_base, participant, "metrics.csv")
        cpu_csv  = os.path.join(metrics_base,  participant, "metrics.csv")
        has_f1   = os.path.isfile(test_csv)
        has_cpu  = os.path.isfile(cpu_csv)
        if has_f1 or has_cpu:
            items.append({
                "participant": participant,
                "f1": test_csv   if has_f1  else None,
                "cpu": cpu_csv   if has_cpu else None,
                "bytes": cpu_csv if has_cpu else None
            })
    return items

def create_f1_figure(path):
    df = pd.read_csv(path)
    df["Round"] = df.index
    df_long = df[["Round", "TestEpoch/F1Score"]].melt(
        id_vars="Round", var_name="Metric", value_name="F1Score"
    )
    fig = px.line(
        df_long,
        x="Round",
        y="F1Score",
        color="Metric",
        title=f"F1 Score — {os.path.basename(os.path.dirname(path))}",
        labels={"Round": "Round", "F1Score": "F1 Score"}
    )
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig

def create_cpu_figure(path):
    df = pd.read_csv(path)
    df["Time"] = df.index
    fig = px.line(
        df,
        x="Time",
        y="Resources/CPU_percent",
        title=f"CPU Usage (%) — {os.path.basename(os.path.dirname(path))}",
        labels={"Time": "Time", "Resources/CPU_percent": "CPU %"},
        markers=True
    )
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig

def create_bytes_figure(path):
    df = pd.read_csv(path)
    df["Time"] = df.index
    fig = px.line(
        df,
        x="Time",
        y="Resources/Bytes_sent",
        title=f"Bytes Sent — {os.path.basename(os.path.dirname(path))}",
        labels={"Time": "Time", "Resources/Bytes_sent": "Bytes Sent"},
        markers=True
    )
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig

app.layout = html.Div(
    style={'backgroundColor': colors['background'], 'padding': '20px'},
    children=[
        html.H1("Metrics Dashboard", style={'textAlign': 'center', 'color': colors['text']}),
        html.Div(id='loading-message', style={'textAlign': 'center', 'color': colors['text']}),
        dcc.Interval(id='interval', interval=5000, n_intervals=0),
        html.Div(
            id='graphs-container',
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '40px',
                'paddingTop': '20px'
            }
        )
    ]
)

@app.callback(
    Output('graphs-container', 'children'),
    Output('loading-message', 'children'),
    Input('interval', 'n_intervals')
)
def update_graphs(n):
    participants = get_participant_paths()
    if not participants:
        return [], 'Waiting for data...'

    f1_graphs    = []
    cpu_graphs   = []
    bytes_graphs = []

    for i, p in enumerate(participants):
        # F1 Score graphs
        if p['f1']:
            f1_graphs.append(
                dcc.Graph(id=f'graph-f1-{i}', figure=create_f1_figure(p['f1']))
            )
        else:
            f1_graphs.append(
                html.Div("No F1 Score data available",
                         style={'color': colors['text'], 'textAlign': 'center'})
            )

        # CPU Usage graphs
        if p['cpu']:
            cpu_graphs.append(
                dcc.Graph(id=f'graph-cpu-{i}', figure=create_cpu_figure(p['cpu']))
            )
        else:
            cpu_graphs.append(
                html.Div("No CPU data available",
                         style={'color': colors['text'], 'textAlign': 'center'})
            )

        # Bytes Sent graphs
        if p['bytes']:
            bytes_graphs.append(
                dcc.Graph(id=f'graph-bytes-{i}', figure=create_bytes_figure(p['bytes']))
            )
        else:
            bytes_graphs.append(
                html.Div("No bytes sent data available",
                         style={'color': colors['text'], 'textAlign': 'center'})
            )

    # Row for F1 Score
    f1_row = html.Div(
        f1_graphs,
        style={
            'display': 'flex',
            'flexDirection': 'row',
            'flexWrap': 'nowrap',
            'overflowX': 'auto',
            'gap': '30px'
        }
    )

    # Row for CPU Usage
    cpu_row = html.Div(
        cpu_graphs,
        style={
            'display': 'flex',
            'flexDirection': 'row',
            'flexWrap': 'nowrap',
            'overflowX': 'auto',
            'gap': '30px'
        }
    )

    # Row for Bytes Sent
    bytes_row = html.Div(
        bytes_graphs,
        style={
            'display': 'flex',
            'flexDirection': 'row',
            'flexWrap': 'nowrap',
            'overflowX': 'auto',
            'gap': '30px'
        }
    )

    return [f1_row, cpu_row, bytes_row], 'Data loaded.'

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
