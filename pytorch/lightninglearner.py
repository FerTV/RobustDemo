#
# This file is an adaptation and extension of the p2pfl library (https://pypi.org/project/p2pfl/).
# Refer to the LICENSE file for licensing information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import logging
import os
import pickle
from collections import OrderedDict

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import CSVLogger

from pytorch.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from pytorch.learning.learner import NodeLearner


###########################
#    LightningLearner     #
###########################


class LightningLearner(NodeLearner):
    """
    PyTorch Lightning-based Learner.

    Attributes:
        model: Model to be trained.
        data: Data for train/val/test.
        epochs: Number of epochs to train.
        train_logger, val_logger, test_logger: Separate CSVLoggers.
    """

    def __init__(self, model, data, config=None, logger=None):
        self.model = model
        # self.model = torch.compile(model)  # PyTorch 2.0 if desired
        self.data = data
        self.config = config

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

        self.epochs = 1
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

        # Federated Learning information
        self.round = 0
        self.local_step = 0
        self.global_step = 0

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data

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
            torch.save(self.get_parameters(), path)
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

    def fit(self):
        """
        Trains the model (training only) and then validates,
        saving metrics separately in train/ and val/.
        """
        if self.epochs <= 0:
            return

        try:
            # 1) Trainer for training — only train_logger, without validation
            trainer_train = self._build_trainer([self.train_logger], disable_val=True)
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
        if hasattr(self.train_logger, 'local_step'):
            setattr(self.train_logger, 'local_step', 0)
        self.round += 1
