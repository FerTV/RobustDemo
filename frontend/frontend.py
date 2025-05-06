from datetime import datetime
import glob
import io
import multiprocessing
import os
import subprocess
import zipfile
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

import httpx

# ---- Scenario class ----
class Scenario:
    def __init__(
        self,
        topology,
        nodes,
        n_nodes,
        dataset,
        iid,
        model,
        agg_algorithm,
        rounds,
        accelerator,
        network_subnet,
        network_gateway,
        epochs,
    ):
        self.topology = topology
        self.nodes = nodes
        self.n_nodes = n_nodes
        self.dataset = dataset
        self.iid = iid
        self.model = model
        self.agg_algorithm = agg_algorithm
        self.rounds = rounds
        self.logginglevel = True
        self.accelerator = accelerator
        self.network_subnet = network_subnet
        self.network_gateway = network_gateway
        self.epochs = epochs
        
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# ---- Directories setup ----
config_dir = os.path.join(os.getcwd(), "robust", "config")
log_dir    = os.path.join(os.getcwd(), "robust", "logs")
models_dir = os.path.join(os.getcwd(), "robust", "models")

os.makedirs(config_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

scenario_path = os.path.join(os.getcwd(), "scenario.json")

# ---- Initialize FastAPI app ----
frontend = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

# ---- FastAPI routes ----
@frontend.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@frontend.post("/robust/run/scenario")
async def set_scenario(request: Request):
    try:
        date = datetime.strftime(datetime.now(), "%d_%m_%Y_%H_%M_%S")
        base_dir = os.path.dirname(__file__)
        app_path = os.path.join(base_dir, "app.py")
        process = subprocess.Popen(["python3", app_path], env={**os.environ, "DATE": date}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        
        project_root = os.path.dirname(base_dir)
        dash_path = os.path.join(project_root, "dash.txt")
        with open(dash_path, "w") as f:
            f.write(str(process.pid))
    except Exception as e:
        print(f"Error launching dash {e}")
    
    data = await request.json()
    data["date"] = date 
    
    async with httpx.AsyncClient() as client:
       resp = await client.post(
           "http://127.0.0.1:8000/api/robust/run/scenario",
           json=data
       )
    return JSONResponse(content=resp.json())


@frontend.post("/robust/stop/scenario")
async def stop_scenario(): 
    async with httpx.AsyncClient() as client:
       resp = await client.post(
           "http://127.0.0.1:8000/api/robust/stop/scenario"
       )
    return JSONResponse(content=resp.json())

@frontend.get("/robust/models")
async def get_models(
    scenario_date: str,
    participant_id: Optional[int] = None,
    round: Optional[int] = None
):
    model_path = os.path.join(models_dir, scenario_date)
    if not os.path.exists(model_path):
        return JSONResponse({"error": "Scenario not found"}, status_code=404)

    files_to_return = []
    for file in os.listdir(model_path):
        if participant_id is not None and round is not None:
            if file == f"participant_{participant_id}_round_{round}_model.pth":
                return FileResponse(
                    os.path.join(model_path, file),
                    media_type="application/octet-stream",
                    filename=file
                )
        elif participant_id is not None and f"participant_{participant_id}_" in file:
            files_to_return.append(file)
        elif round is not None and f"_round_{round}_" in file:
            files_to_return.append(file)
        elif participant_id is None and round is None:
            files_to_return.append(file)

    if not files_to_return:
        return JSONResponse({"error": "No models found"}, status_code=404)

    if len(files_to_return) == 1:
        file = files_to_return[0]
        return FileResponse(
            os.path.join(model_path, file),
            media_type="application/octet-stream",
            filename=file
        )
    
    # Zip multiple model files
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in files_to_return:
            zip_file.write(
                os.path.join(model_path, file),
                arcname=file
            )
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=models.zip"}
    )

@frontend.get("/robust/metrics")
async def get_metrics(
    scenario_date: str,
    participant_id: Optional[int] = None
):
    metrics_path = os.path.join(log_dir, scenario_date, "metrics")
    if not os.path.exists(metrics_path):
        return JSONResponse({"error": "Scenario not found"}, status_code=404)

    if participant_id is not None:
        metrics_file = os.path.join(
            metrics_path, f"participant_{participant_id}", "metrics.csv"
        )
        if os.path.exists(metrics_file):
            return FileResponse(
                metrics_file,
                media_type="application/octet-stream",
                filename=f"metrics_participant_{participant_id}.csv"
            )
        return JSONResponse(
            {"error": "Metrics not found for participant"},
            status_code=404
        )
    
    # Collect all participants' metrics
    files_to_return = []
    for participant_folder in os.listdir(metrics_path):
        participant_metrics = os.path.join(
            metrics_path, participant_folder, "metrics.csv"
        )
        if os.path.exists(participant_metrics):
            files_to_return.append((participant_folder, participant_metrics))

    if not files_to_return:
        return JSONResponse({"error": "No metrics found"}, status_code=404)
    if len(files_to_return) == 1:
        pf, path = files_to_return[0]
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"metrics_{pf}.csv"
        )
    # Zip multiple metrics files
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for pf, path in files_to_return:
            zip_file.write(path, arcname=f"metrics_{pf}.csv")
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=all_metrics.zip"}
    )

@frontend.get("/dash")
async def redirect_to_dash():
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.get("http://localhost:8050/dash/")
        return RedirectResponse("http://localhost:8050/dash/")
    except httpx.RequestError:
        html_content = """
        <html>
            <head>
                <title>Dash not available</title>
            </head>
            <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 10%;">
                <h1 style="color: #ff5555;">No experiment running</h1>
                <p>Please start the Dash server to view the dashboard</p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=503)

# ---- Entrypoint ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app=frontend,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
