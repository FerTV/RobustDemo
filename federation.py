import logging
import sys
import time

from config import Config
from pytorch.mnist.mnist import MNISTDataModule
from pytorch.mnist.models.mlp import MNISTModelMLP
from node import Node


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'  
    )

    config_path = str(sys.argv[1])
    config = Config(entity="participant", participant_config_file=config_path)

    n_nodes = config.participant["scenario_args"]["n_nodes"]
    experiment_name = config.participant["scenario_args"]["name"]
    model_name = config.participant["model_args"]["model"]
    idx = config.participant["device_args"]["idx"]
    host = config.participant["network_args"]["ip"]
    port = config.participant["network_args"]["port"]
    neighbors = config.participant["network_args"]["neighbors"].split()

    rounds = config.participant["scenario_args"]["rounds"]
    epochs = config.participant["training_args"]["epochs"]

    aggregation_algorithm = config.participant["aggregator_args"]["algorithm"]

    dataset = config.participant["data_args"]["dataset"]
    model = None
    
    if dataset == "MNIST":
        dataset = MNISTDataModule(sub_id=idx, number_sub=n_nodes, iid=True)
        if model_name == "MLP":
            model = MNISTModelMLP()
            
    if aggregation_algorithm == "FedAvg":
        pass
    else:
        raise ValueError(f"Aggregation algorithm {aggregation_algorithm} not supported")
    
    node = Node(
        idx=idx,
        experiment_name=experiment_name,
        model=model,
        data=dataset,
        host=host,
        port=port,
        config=config,
        encrypt=False
    )
    
    node.start()
    print("Node started, grace time for network start-up (30s)")
    time.sleep(20)  # Wait for the participant to start and register in the network

    # Node Connection to the neighbors
    for i in neighbors:
        print(f"Connecting to {i}")
        node.connect_to(i.split(':')[0], int(i.split(':')[1]), full=False)
        time.sleep(5)

    logging.info(f"Neighbors: {node.get_neighbors()}")
    logging.info(f"Network nodes: {node.get_network_nodes()}")
    
    start_node = config.participant["device_args"]["start"]

    if start_node:
        node.set_start_learning(rounds=rounds, epochs=epochs)  # rounds=10, epochs=5

if __name__ == "__main__":
    main()