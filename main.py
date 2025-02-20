from datetime import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time

from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi import Request

from role import Role

class Scenario():
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

# Globals
config_dir = os.path.join(os.getcwd(), "robust", "config")
log_dir = os.path.join(os.getcwd(), "robust", "logs")
models_dir = os.path.join(os.getcwd(), "robust", "models")

os.makedirs(config_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

scenario_path = os.path.join(os.getcwd(), "scenario.json")

# App control
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'  
)

def signal_handler(signal, frame):
    stop_participants()
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

# Initialize FastAPI app
app = FastAPI()

@app.post("/robust/run/scenario")
async def set_scenario(request: Request):
    scenario = await request.json()
    try:
        with open(scenario_path, "w") as f:
            json.dump(scenario, f, indent=4)
            
        await create_configs()
        return {"message": "Scenario running successfully"}
    except Exception as e:
        return {"error": f"Failed to run scenario: {str(e)}"}
    
@app.get("/roubust/models")
async def get_models(scenario_date, particpant_id, round):
    model = os.path.join(models_dir, scenario_date, f"participant_{particpant_id}_round_{round}_model.pth")
    
    if not os.path.exists(model):
        return {"error": "Model not found"}
    
    return FileResponse(model, media_type="application/octet-stream", filename=f"participant_{particpant_id}_round_{round}_model.pth")
    
# Basics
def stop_participants():
        if sys.platform == "win32":
            try:
                # kill all the docker containers which contain the word "nebula-core"
                commands = [
                    """docker kill $(docker ps -q --filter ancestor=robust) | Out-Null""",
                    """docker rm $(docker ps -a -q --filter ancestor=robust) | Out-Null""",
                    """docker network rm $(docker network ls | Where-Object { ($_ -split '\s+')[1] -like 'robust_network' } | ForEach-Object { ($_ -split '\s+')[0] }) | Out-Null""",
                ]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(f'powershell.exe -Command "{command}"')
                    # logging.info(f"Windows Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))
        else:
            try:
                commands = [
                    """docker kill $(docker ps -q --filter ancestor=robust) > /dev/null 2>&1""",
                    """docker rm $(docker ps -a -q --filter ancestor=robust) > /dev/null 2>&1""",
                    """docker network rm $(docker network ls | grep robust_network | awk '{print $1}') > /dev/null 2>&1""",
                ]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(command)
                    # logging.info(f"Linux Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))

# Scenarios  
def gen_topology(scenario):
    topology = None 
    if scenario.topology == "Fully":
        fully_connected = np.array(
            nx.to_numpy_array(nx.watts_strogatz_graph(scenario.n_nodes, scenario.n_nodes - 1, 0)),
            dtype=np.float32,
        )
        np.fill_diagonal(fully_connected, 0)

        for i in range(scenario.n_nodes):
            for j in range(scenario.n_nodes):
                if fully_connected[i][j] != 1:
                    fully_connected[i][j] = 1

        np.fill_diagonal(fully_connected, 0)
        
        topology = fully_connected
    else:
        ring = np.array(
            nx.to_numpy_array(nx.watts_strogatz_graph(scenario.n_nodes, 2, 0)), dtype=np.float32
        )

        # Create random links between nodes in topology_ring
        for i in range(scenario.n_nodes):
            for j in range(scenario.n_nodes):
                if ring[i][j] == 0:
                    if random.random() < 0.1:
                        ring[i][j] = 1
                        ring[j][i] = 1

        np.fill_diagonal(ring, 0)
        topology = ring
        
    return topology

def draw_graph(plot=False, path=None, scenario=Scenario, topology=None):
        g = nx.from_numpy_array(topology)
        pos = nx.spring_layout(g, k=0.15, iterations=20, seed=42)

        fig = plt.figure(num="Network topology", dpi=100, figsize=(6, 6), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])

        labels = {}
        color_map = []
        server = False
        for k, node in scenario.nodes.items():
            if node['role'] == Role.AGGREGATOR:
                color_map.append("orange")
            elif node['role'] == Role.PROXY:
                color_map.append("purple")
            else:
                color_map.append("red")
            if node['ip'] is not None and node['ip'] != "127.0.0.1":
                labels[int(k)] = f"P{k}\n" + node['ip'] + ":" + node['port']
            else:
                labels[int(k)] = f"P{k}\n" + "localhost" + ":" + node['port']

        nx.draw_networkx_nodes(g, pos, node_color=color_map, linewidths=2)
        nx.draw_networkx_labels(g, pos, labels, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(g, pos, width=2)

        roles = [node['role'] for k, node in scenario.nodes.items()]
        if Role.AGGREGATOR in roles:
            plt.scatter([], [], c="orange", label='Aggregator')
        if Role.PROXY in roles:
            plt.scatter([], [], c="purple", label='Proxy')
        if Role.IDLE in roles:
            plt.scatter([], [], c="red", label='Idle')

        plt.legend()
        plt.savefig(f"{path}", dpi=100, bbox_inches="tight", pad_inches=0)

        if plot:
            plt.show()

def get_neighbors(scenario, topology, idx):
    neighbors_index = []
    neighbors_data = []
    for i in range(scenario.n_nodes):
        if topology[idx][i] == 1:
            neighbors_index.append(i)
            neighbors_data.append(scenario.nodes[str(i)])
    neighbors_data_string = ""
    for i in neighbors_data:
        neighbors_data_string += str(i['ip']) + ":" + str(i['port'])
        if neighbors_data[-1] != i:
            neighbors_data_string += " "
    return neighbors_data_string

async def create_configs():    
    date = datetime.strftime(datetime.now(), "%d_%m_%Y_%H_%M_%S")
    config_dir = os.path.join(os.getcwd(), "robust", "config", date)
    log_dir = os.path.join(os.getcwd(), "robust", "logs", date)
    models_dir = os.path.join(os.getcwd(), "robust", "models", date)
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    logging.info("configs created")

    with open(scenario_path, "r", encoding="utf-8") as file:
        scenario_json = json.load(file)
        
    scenario = Scenario.from_dict(scenario_json)
    topology = gen_topology(scenario)
    participants = []
    idx_start_node = 0
        
    # Save node settings
    for node in scenario.nodes:
        node_config = scenario.nodes[node]
        participant_file = os.path.join(config_dir, f'participant_{node_config["id"]}.json')
        os.makedirs(os.path.dirname(participant_file), exist_ok=True)
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "participant.json.txt"),
            participant_file,
        )
        
        with open(participant_file) as f:
            participant_config = json.load(f)
        
        participant_config["network_args"]["ip"] = node_config["ip"]
        participant_config["network_args"]["port"] = int(node_config["port"])
        participant_config['network_args']['neighbors'] = get_neighbors(scenario, topology, node_config["id"])
        participant_config["device_args"]["idx"] = node_config["id"]
        participant_config['device_args']['uid'] = hashlib.sha1((str(participant_config["network_args"]["ip"]) + str(participant_config["network_args"]["port"])).encode()).hexdigest()
        participant_config["device_args"]["start"] = node_config["start"]
        participant_config["device_args"]["role"] = node_config["role"]
        participant_config["device_args"]["proxy"] = node_config["proxy"]
        participant_config["device_args"]["malicious"] = node_config["malicious"]
        participant_config["scenario_args"]["n_nodes"] = int(scenario.n_nodes)
        participant_config["scenario_args"]["rounds"] = int(scenario.rounds)
        participant_config["data_args"]["dataset"] = scenario.dataset
        participant_config["data_args"]["iid"] = scenario.iid
        participant_config["model_args"]["model"] = scenario.model
        participant_config["training_args"]["epochs"] = int(scenario.epochs)
        participant_config["device_args"]["accelerator"] = scenario.accelerator
        participant_config["device_args"]["logging"] = True
        participant_config["aggregator_args"]["algorithm"] = scenario.agg_algorithm
        with open(participant_file, "w") as f:
            json.dump(participant_config, f, sort_keys=False, indent=2)
            
        if node_config["start"] == True:
            idx_start_node = node_config["id"]
            
        participants.append(participant_config)
        
    draw_graph(path=f"{log_dir}/topology.png", plot=False, scenario=scenario, topology=topology)
        
    start_federation(idx_start_node, participants, date)

def start_federation(idx_start_node, participants, date):
    docker_compose_template = """
    services:
    {}
    """
    participant_template = """
              participant{}:
                image: {}
                volumes:
                    - {}:/robust
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                      ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.11 /robust/federation.py {}
                depends_on:
                    - participant{}
                networks:
                    robust:
                        ipv4_address: {}
            """
    participant_template_start = """
              participant{}:
                image: {}
                volumes:
                    - {}:/robust
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                      ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.11 /robust/federation.py {}
                networks:
                    robust:
                        ipv4_address: {}
            """
    network_template = """
    networks:
        robust:
            name: robust
            driver: bridge
            ipam:
              config:
                - subnet: 192.168.50.0/24
                  gateway: 192.168.50.1
    """
    # Generate the Docker Compose file dynamically
    services = ""
    #nodes.sort(key=lambda x: x['device_args']['idx'])
    for node in participants:
        idx = node['device_args']['idx']
        path = f"/robust/robust/config/{date}/participant_{idx}.json"
        logging.info("Starting node {} with configuration {}".format(idx, path))
        logging.info("Node {} is listening on ip {}".format(idx, node['network_args']['ip']))
        # Add one service for each participant
        if idx != idx_start_node:
            services += participant_template.format(idx,
                                                    "robust" if node['device_args']['accelerator'] == "cpu" else "robust-gpu",
                                                    os.getcwd(),
                                                    "192.168.50.1",
                                                    path,
                                                    idx_start_node,
                                                    node['network_args']['ip'])
        else:
            services += participant_template_start.format(idx,
                                                          "robust" if node['device_args']['accelerator'] == "cpu" else "robust-gpu",
                                                          os.getcwd(),
                                                          "192.168.50.1",
                                                          path,
                                                          node['network_args']['ip'])
    docker_compose_file = docker_compose_template.format(services)
    docker_compose_file += network_template.format("192.168.50.0/24", "192.168.50.1")
    # Write the Docker Compose file in config directory
    with open(f"{config_dir}/{date}/docker-compose.yml", "w") as f:
        f.write(docker_compose_file)
    for node in participants:
        node['tracking_args']['log_dir'] = f"/robust/robust/logs/{date}"
        node['tracking_args']['config_dir'] = f"/robust/robust/config/{date}"
        node['tracking_args']['models_dir'] = f"/robust/robust/models/{date}"
        # Write the config file in config directory
        with open(f"{config_dir}/{date}/participant_{node['device_args']['idx']}.json", "w") as f:
            json.dump(node, f, indent=4)
    # Start the Docker Compose file, catch error if any
    try:
        subprocess.check_call(["docker", "compose", "-f", f"{config_dir}/{date}/docker-compose.yml", "up", "-d"])
    except subprocess.CalledProcessError as e:
        logging.error("Docker Compose failed to start, please check if Docker is running and Docker Compose is installed.")
        logging.error(e)
        raise e

def main():
    logging.info("STARTING...")
    # create_configs()
    print("Press CTRL+C to exit")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()