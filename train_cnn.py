from comet_ml import Experiment
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CometLogger

from scrabblebot_framework.board_value_cnn import BoardValueCNN


class ScrabbleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    # comet_logger = CometLogger(
    #     api_key="Wj297ZgqXDxiYjjIRFrtGFQ6K",
    # )
    # # Force logger to init experiment for live logging
    # exp = comet_logger.experiment
    model = BoardValueCNN({"lr": 1e-4})
    train_dataset = ScrabbleDataset(
        "./cnn_data/60k/cnn_train_data.npy", "./cnn_data/60k/cnn_train_labels.npy")
    test_dataset = ScrabbleDataset(
        "./cnn_data/60k/cnn_test_data.npy", "./cnn_data/60k/cnn_test_labels.npy")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True)
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="./cnn_checkpoints",
        max_epochs=10000,
        # logger=comet_logger
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader)
    # trainer.test(model=model, dataloaders=test_dataloader)
