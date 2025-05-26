import logging
import sys
import time

from config import Config
from pytorch.datamodule import DataModule
from pytorch.mnist.mnist import MNISTDataset
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

    #target_changed_label = config.participant["adversarial_args"]["target_changed_label"]
    #noise_type = config.participant["adversarial_args"]["noise_type"]
    #num_workers = config.participant["data_args"]["num_workers"]

    # Config of attacks
    attacks = config.participant["device_args"]["attack"]
    poisoned_ratio = config.participant["adversarial_args"]["poisoned_ratio"]
    poisoned_persent = config.participant["adversarial_args"]["poisoned_sample_percent"]
    targeted = str(config.participant["adversarial_args"]["targeted"])
    target_label = config.participant["adversarial_args"]["target_label"]
    target_changed_label = config.participant["adversarial_args"]["target_changed_label"]
    noise_type = config.participant["adversarial_args"]["noise_type"]
    data_poisoning = False

    # config of attacks
    if attacks == "Label Flipping":
        label_flipping = True
        poisoned_ratio = 0
        if targeted == "true" or targeted == "True":
            targeted = True
        else:
            targeted = False
    else:
        label_flipping = False

    indices_dir = config.participant["tracking_args"]["models_dir"]
    num_workers = 15
    target_changed_label = 0
    noise_type = "salt"

    dataset = config.participant["data_args"]["dataset"]
    iid = config.participant["data_args"]["iid"]
    model = None
    
    if dataset == "MNIST":
        dataset = MNISTDataset(num_classes=10, sub_id=idx, number_sub=n_nodes, iid=iid, partition="percent", seed=42, config=config)
        if model_name == "MLP":
            model = MNISTModelMLP()

    dataset = DataModule(train_set=dataset.train_set, train_set_indices=dataset.train_indices_map, test_set=dataset.test_set, test_set_indices=dataset.test_indices_map, num_workers=num_workers, sub_id=idx, number_sub=n_nodes, indices_dir=indices_dir, label_flipping=label_flipping, data_poisoning=data_poisoning, poisoned_persent=poisoned_persent, poisoned_ratio=poisoned_ratio, targeted=targeted, target_label=target_label,
                         target_changed_label=target_changed_label, noise_type=noise_type)
            
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