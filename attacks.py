import copy
import random
from typing import Any
import torch
import numpy as np
import logging

# Set log level for specific libraries to reduce verbosity
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

def create_attack(attack_name):
    """
    Factory function to create an attack object based on its name.
    
    Args:
        attack_name (str): The name of the attack to create (e.g., 'Model Poisoning').
    
    Returns:
        Attack: An attack object of the corresponding attack type.
    
    Raises:
        ValueError: If the attack name is not supported.
    """
    if attack_name == "Model Poisoning":
        return ModelPoisoning()
    else:
        raise ValueError(f"Attack {attack_name} not supported")


class Attack:
    """
    Base class for attacks, allowing polymorphism for different attack types.
    """
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Calls the `attack` method to perform the attack.
        
        Args:
            *args: Arguments passed to the attack method.
            **kwds: Keyword arguments passed to the attack method.
        
        Returns:
            Any: The result of the attack.
        """
        return self.attack(*args, **kwds)

    def attack(self, received_weights):
        """
        Abstract method that should be overridden in subclasses to perform a specific attack.
        
        Args:
            received_weights (dict): The model weights to which the attack will be applied.
        
        Returns:
            dict: The attacked weights.
        
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class ModelPoisoning(Attack):
    """
    Class that performs a model poisoning attack by injecting noise into model weights.
    """
    
    def __init__(self, strength=10000, perc=1.0):
        """
        Initializes the ModelPoisoning attack object with given parameters.
        
        Args:
            strength (float): The strength of the noise to inject. Default is 10000.
            perc (float): The percentage of layers to affect. Default is 1.0 (all layers).
        """
        super().__init__()
        self.strength = strength
        self.perc = perc

    def attack(self, received_weights):
        """
        Performs noise injection on the received model weights.
        
        Args:
            received_weights (dict): The model weights to which the noise will be added.
        
        Returns:
            dict: The poisoned model weights with noise added.
        """
        logging.info("[ModelPoisoning] Performing noise injection attack")
        
        # Get the list of layers in the model
        lkeys = list(received_weights.keys())
        
        # Inject noise into each layer
        for k in lkeys:
            logging.info(f"Layer noised: {k}")
            received_weights[k].data += torch.randn(received_weights[k].shape) * self.strength
        
        return received_weights


class LabelFlipping(Attack):
    """
    Class that performs a label flipping attack on the dataset by changing the labels of some data points.
    """
    
    def __init__(self):
        """
        Initializes the LabelFlipping attack object.
        """
        super().__init__()

    def labelFlipping(self, dataset, indices, poisoned_persent=0, targeted=False, target_label=4, target_changed_label=7):
        """
        Flips the labels of a subset of the dataset. Can either flip labels randomly or perform a targeted label change.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset containing the training data.
            indices (list): List of indices of the data points to be attacked.
            poisoned_persent (float): Percentage of labels to flip (randomly). Default is 0.
            targeted (bool): Whether to perform a targeted attack. Default is False (random label flipping).
            target_label (int): The label to be targeted in a targeted attack. Default is 4.
            target_changed_label (int): The new label to which the targeted label will be changed. Default is 7.
        
        Returns:
            torch.utils.data.Dataset: The modified dataset with flipped labels.
        """
        logging.info("[LabelFlipping] Performing label flipping attack")
        
        # Create a deep copy of the dataset to avoid modifying the original
        new_dataset = copy.deepcopy(dataset)
        
        # Ensure the dataset's labels are stored as a numpy array for easy manipulation
        if not isinstance(new_dataset.targets, np.ndarray):
            new_dataset.targets = np.array(new_dataset.targets)
        else:
            new_dataset.targets = new_dataset.targets.copy()

        # Perform random label flipping
        if not targeted:
            num_indices = len(indices)
            num_flipped = int(poisoned_persent * num_indices)
            
            # If there are no indices or the number of flipped labels exceeds the available indices, return
            if num_indices == 0 or num_flipped > num_indices:
                return new_dataset
            
            flipped_indices = random.sample(indices, num_flipped)
            class_list = list(set(new_dataset.targets.tolist()))
            
            # Flip labels randomly for the selected indices
            for i in flipped_indices:
                current_label = new_dataset.targets[i]
                new_label = random.choice(class_list)
                
                # Ensure the new label is different from the current label
                while new_label == current_label:
                    new_label = random.choice(class_list)
                
                new_dataset.targets[i] = new_label
        else:
            # Perform targeted label flipping
            for i in indices:
                if int(new_dataset.targets[i]) == target_label:
                    new_dataset.targets[i] = target_changed_label
        
        return new_dataset
