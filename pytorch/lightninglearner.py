import logging
import os
import pickle
from collections import OrderedDict
import random
import traceback
import hashlib
import numpy as np
import io
import gzip

from lightning.pytorch.loggers import CSVLogger
#from attacks import labelFlipping
from pytorch.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from pytorch.learning.learner import NodeLearner
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import copy

from torch.nn import functional as F

###########################
#    LightningLearner     #
###########################


class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None, logger=None):
        # logging.info("[Learner] Compiling model... (BETA)")
        # self.model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.__trainer = None
        self.epochs = 1
        logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

        # If a single CSVLogger is provided, split it into train/val/test
        if isinstance(logger, CSVLogger):
            base_dir = getattr(logger, "save_dir", None) or getattr(logger, "root_dir", None)
            version = getattr(logger, "version", None)
            self.train_logger = CSVLogger(save_dir=base_dir, name="train", version=version)
            self.val_logger   = CSVLogger(save_dir=base_dir, name="val",   version=version)
            self.test_logger  = CSVLogger(save_dir=base_dir, name="test",  version=version)
        else:
            # If separate loggers are already provided, assume logger is a list
            self.train_logger = logger if isinstance(logger, list) and len(logger) >= 1 else logger
            self.val_logger   = logger if isinstance(logger, list) and len(logger) >= 2 else None
            self.test_logger  = logger if isinstance(logger, list) and len(logger) >= 3 else None

        # FL information
        self.round = 0
        
        self.fix_randomness()
        # self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)
        

    def fix_randomness(self):
        seed = self.config.participant["scenario_args"]["random_seed"]
        logging.info("[Learner] Fixing randomness with seed {}".format(seed))
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def get_round(self):
        return self.round

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data

    ####
    # Model weights
    # Encode/decode parameters: https://pytorch.org/docs/stable/notes/serialization.html
    # There are other ways to encode/decode parameters: protobuf, msgpack, etc.
    ####
    def encode_parameters(self, params=None, contributors=None, weight=None):
        if params is None:
            params = self.model.state_dict()
        array = [val.cpu().numpy() for _, val in params.items()]
        return pickle.dumps((array, contributors, weight))

    def decode_parameters(self, data):
        try:
            params, contributors, weight = pickle.loads(data)
            params_dict = zip(self.model.state_dict().keys(), params)
            return (
                OrderedDict({k: torch.tensor(v) for k, v in params_dict}),
                contributors,
                weight,
            )
        except Exception:
            raise DecodingParamsError("Error decoding parameters")

    def check_parameters(self, params):
        if set(params.keys()) != set(self.model.state_dict().keys()):
            return False
        for key, value in params.items():
            if value.shape != self.model.state_dict()[key].shape:
                return False
        return True

    def set_parameters(self, params):
        try:
            self.model.load_state_dict(params)
        except Exception:
            raise ModelNotMatchingError("Model parameters do not match")

    def get_parameters(self):
        return self.model.state_dict()

    def save_model(self, round):
        try:
            idx = self.config.participant["device_args"]["idx"]
            path = os.path.join(
                self.config.participant["tracking_args"]["models_dir"],
                f"participant_{idx}_round_{round}_model.pth"
            )
            torch.save(self.model, path)
            logging.info(f"Model saved at: {path} (round {round})")
        except Exception as e:
            logging.error(f"Error saving the model: {e}")

    def set_epochs(self, epochs):
        self.epochs = epochs

    def _build_trainer(self, loggers, disable_val: bool = False):
        """
        Creates a Trainer with the specified callbacks and loggers.
        If disable_val=True, passes limit_val_batches=0 to skip validation in fit().
        """
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            ),
            leave=True,
        )
        trainer_kwargs = dict(
            callbacks=[RichModelSummary(max_depth=1), progress_bar],
            max_epochs=self.epochs,
            accelerator=self.config.participant["device_args"]["accelerator"],
            devices="auto",
            logger=loggers,
            log_every_n_steps=20,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
        )

        if disable_val:
            # Prevent any validation during fit()
            trainer_kwargs['limit_val_batches'] = 0

        return Trainer(**trainer_kwargs)

    def fit(self, label_flipping=False, poisoned_persent=0, targeted=False, target_label=4, target_changed_label=7):    
        """
        Trains the model (training only) and then validates,
        saving metrics separately in train/ and val/.
        """
        if self.epochs <= 0:
            return

        try:
            logging.info(f"[Learner] label_flipping: {label_flipping}")
            # 1) Trainer for training — only train_logger, without validation
            trainer_train = self._build_trainer([self.train_logger], disable_val=True)
            if label_flipping:
                # self.data.train_set = labelFlipping(
                #             self.data.train_set, 
                #             self.data.train_set_indices, 
                #             80,
                #             False,
                #             target_label=target_label,
                #             target_changed_label=target_changed_label) 
                pass
            trainer_train.fit(self.model, self.data)

            # 2) Trainer for validation — only val_logger
            trainer_val = self._build_trainer([self.val_logger], disable_val=False)
            trainer_val.validate(self.model, self.data, verbose=True)

        except Exception as e:
            logging.error(f"Error during fit()/validate(): {e}")

    def evaluate(self):
        """
        Tests the model, logging only to test_logger.
        """
        if self.epochs <= 0:
            return None
        try:
            trainer = self._build_trainer([self.test_logger])
            trainer.test(self.model, self.data, verbose=True)
        except Exception as e:
            logging.error(f"Error during test(): {e}")
            return None

    def interrupt_fit(self):
        # Interrupts are not handled in this design
        pass

    def get_num_samples(self):
        return (
            len(self.data.train_dataloader().dataset),
            len(self.data.test_dataloader().dataset),
        )

    def init(self):
        self.close()

    def close(self):
        # Clean up if needed when finishing
        pass

    def finalize_round(self):
        # if hasattr(self.train_logger, 'local_step'):
        #     setattr(self.train_logger, 'local_step', 0)
        self.round += 1
