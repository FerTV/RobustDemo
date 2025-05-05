import glob
import os
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import pandas as pd

# Asegurarse de que la variable de entorno DATE
# esté definida
date = os.getenv("DATE")
if not date:
    raise ValueError("The DATE environment variable is not set. Please set it before running the app.")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix="/dash/",
    routes_pathname_prefix="/dash/"
)

colors = {'background': '#111111', 'text': '#7FDBFF'}

def get_csv_paths():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path = os.path.join(root_path, "robust", "logs", date, "test")
    csv_paths = []
    if not os.path.exists(base_path):
        return []
    for participant in sorted(os.listdir(base_path)):
        participant_dir = os.path.join(base_path, participant)
        if os.path.isdir(participant_dir):
            csv_file = os.path.join(participant_dir, "metrics.csv")
            if os.path.isfile(csv_file):
                csv_paths.append(csv_file)
    return csv_paths

def create_figure_from_csv(path):
    try:
        df = pd.read_csv(path)
        df["Round"] = df.index
        df_long = df[["Round", "Test/F1Score", "TestEpoch/F1Score"]].melt(
            id_vars="Round", var_name="Type", value_name="F1Score"
        )
        fig = px.line(
            df_long, x="Round", y="F1Score", color="Type",
            title=f"F1-Score - {os.path.basename(os.path.dirname(path))}",
            labels={"Round": "Round", "F1Score": "F1-Score"}
        )
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        return fig
    except Exception as e:
        return px.line(title=f"Error reading {path}: {str(e)}")

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(id="main-title", style={'textAlign': 'center', 'color': colors['text']}), 
    html.Div(id='loading-message', children='Waiting for data...', style={'textAlign': 'center', 'color': colors['text']}), 
    html.Div(children='Comparison of F1-Scores for each participant.', style={'textAlign': 'center', 'color': colors['text']}), 
    dcc.Interval(id='interval', interval=5000, n_intervals=0),  # cada 5s
    html.Div(id='graphs-container')
])

@app.callback(
    Output('main-title', 'children'),
    Output('graphs-container', 'children'),
    Output('loading-message', 'children'),
    Input('interval', 'n_intervals')
)
def update_graphs(n):
    # Obtener las rutas de los archivos CSV
    paths = get_csv_paths()
    
    if not paths:
        # Si no hay archivos CSV, mostramos el mensaje de "Esperando datos"
        title = f'F1-Score - Scenario {date}'
        graphs = []
        loading_message = 'Waiting for data...'
    else:
        # Si hay archivos CSV, generamos los gráficos
        title = f'F1-Score - Scenario {date}'
        graphs = [
            dcc.Graph(id=f'graph-{i}', figure=create_figure_from_csv(p)) for i, p in enumerate(paths)
        ]
        loading_message = 'Data loaded.'
    
    return title, graphs, loading_message

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
