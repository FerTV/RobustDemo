import logging
import os
import csv
import torch  # Para detectar tensores
from pytorch_lightning.loggers import TensorBoardLogger

class CSVLogger(TensorBoardLogger):
    def __init__(self, save_dir, name="default", version=None):
        super().__init__(save_dir, name, version)
        self.csv_file = os.path.join(self.log_dir, "metrics.csv")
        self.local_step = 0
        self.global_step = 0

        self.columns = [
            'Test/Loss', 'Test/Accuracy', 'Test/Precision', 'Test/Recall', 'Test/F1Score',
            'TestEpoch/Accuracy', 'TestEpoch/Precision', 'TestEpoch/Recall', 'TestEpoch/F1Score',
            'epoch', 'step',
            'Train/Loss', 'Train/Accuracy', 'Train/Precision', 'Train/Recall', 'Train/F1Score',
            'Validation/Loss', 'Validation/Accuracy', 'Validation/Precision', 'Validation/Recall', 'Validation/F1Score',
            'ValidationEpoch/Accuracy', 'ValidationEpoch/Precision', 'ValidationEpoch/Recall', 'ValidationEpoch/F1Score',
            'TrainEpoch/Accuracy', 'TrainEpoch/Precision', 'TrainEpoch/Recall', 'TrainEpoch/F1Score'
        ]

        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.columns)
                writer.writeheader()

    def log_metrics(self, metrics, step=None):
        self.local_step = step or 0
        step = self.global_step + self.local_step

        if "epoch" in metrics:
            metrics["epoch"] = int(metrics["epoch"])

        row = {}
        for column in self.columns:
            value = metrics.get(column)
            logging.error(f"Columna: {column}, Valor: {value}, Tipo: {type(value)}")  # Ver tipo de dato

            if isinstance(value, torch.Tensor):
                value = value.item()  # Convertir tensor a float
            row[column] = value
        row["step"] = step  # Registrar el paso global


        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writerow(row)

        super().log_metrics(metrics, step)
