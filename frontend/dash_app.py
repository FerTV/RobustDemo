def run_dash(date):
    import glob
    import os
    import dash_bootstrap_components as dbc
    from dash import Dash, dcc, html, Output, Input
    import plotly.express as px
    import pandas as pd

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        requests_pathname_prefix="/dash/",
        routes_pathname_prefix="/dash/"
    )

    colors = {'background': '#111111', 'text': '#7FDBFF'}

    def get_csv_paths():
        base_path = f"../robust/logs/{date}/test"
        if not os.path.exists(base_path):
            return []
        return sorted(glob.glob(f"{base_path}/participant_*/metrics.csv"))

    def create_figure_from_csv(path):
        try:
            df = pd.read_csv(path)
            df["Round"] = df.index + 1
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
            print(f"Error reading {path}: {str(e)}")
            return px.line(title=f"Error reading {path}: {str(e)}")

    app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.H1(id="main-title", style={'textAlign': 'center', 'color': colors['text']}), 
        html.Div(children='Comparison of F1-Scores for each participant.', style={'textAlign': 'center', 'color': colors['text']}), 
        html.Div(id='graphs-container')
    ])

    @app.callback(
        Output('main-title', 'children'),
        Output('graphs-container', 'children'),
        Input('main-title', 'children')  # Callback se activa solo al iniciar la página
    )
    def update_graphs(_):
        paths = get_csv_paths()
        title = f'F1-Score - Scenario {date}'
        graphs = [dcc.Graph(id=f'graph-{i}', figure=create_figure_from_csv(p)) for i, p in enumerate(paths)]
        return title, graphs

    app.run_server(debug=False, port=8050)
