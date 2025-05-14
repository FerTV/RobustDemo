import copy
import random
from typing import Any
import torch
import numpy as np
from torchmetrics.functional import pairwise_cosine_similarity
from copy import deepcopy
import logging

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

# To take into account:
# - Malicious nodes do not train on their own data
# - Malicious nodes aggregate the weights of the other nodes, but not their own
# - The received weights may be the node own weights (aggregated of neighbors), or
#   if the attack is performed specifically for one of the neighbors, it can take
#   its weights only (should be more effective if they are different).


def create_attack(attack_name):
    """
    Function to create an attack object from its name.
    """
    if attack_name == "Model Poisoning":
        return ModelPoisoning()
    else:
        raise ValueError(f"Attack {attack_name} not supported")


class Attack:

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.attack(*args, **kwds)

    def attack(self, received_weights):
        """
        Function to perform the attack on the received weights. It should return the
        attacked weights.
        """
        raise NotImplementedError

class ModelPoisoning(Attack):
    """
    Function to perform model poisoning attack on the received weights.
    """

    def __init__(self, strength=10000, perc=1.0):
        super().__init__()
        self.strength = strength
        self.perc = perc

    def attack(self, received_weights):
        logging.info("[ModelPoisoning] Performing noise injection attack")
        lkeys = list(received_weights.keys())
        for k in lkeys:
            logging.info(f"Layer noised: {k}")
            received_weights[k].data += torch.randn(received_weights[k].shape) * self.strength
        return received_weights


def labelFlipping(dataset, indices, poisoned_persent=0, targeted=False, target_label=4, target_changed_label=7):
    """
    select flipping_persent of labels, and change them to random values.
    Args:
        dataset: the dataset of training data, torch.util.data.dataset like.
        indices: Indices of subsets, list like.
        flipping_persent: The ratio of labels want to change, float like.
    """
    logging.info("[LabelFlipping] Performing label flipping attack")
    new_dataset = copy.deepcopy(dataset)
    if not isinstance(new_dataset.targets, np.ndarray):
        new_dataset.targets = np.array(new_dataset.targets)
    else:
        new_dataset.targets = new_dataset.targets.copy()

    if not targeted:
        num_indices = len(indices)
        num_flipped = int(poisoned_persent * num_indices)
        if num_indices == 0 or num_flipped > num_indices:
            return
        flipped_indices = random.sample(indices, num_flipped)
        class_list = list(set(new_dataset.targets.tolist()))
        for i in flipped_indices:
            current_label = new_dataset.targets[i]
            new_label = random.choice(class_list)
            while new_label == current_label:
                new_label = random.choice(class_list)
            new_dataset.targets[i] = new_label
    else:
        for i in indices:
            if int(new_dataset.targets[i]) == target_label:
                new_dataset.targets[i] = target_changed_label
    # logging.info(f"[{self.__class__.__name__}] First 20 labels after flipping: {new_dataset.targets[:20]}")
    return new_dataset