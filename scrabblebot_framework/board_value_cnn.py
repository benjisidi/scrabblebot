from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class BoardValueCNN(pl.LightningModule):
    def __init__(self, hparams={"lr": 1e-4}):
        super().__init__()
        """
        Inputs: 3x15x15
        conv1: 3x11x11
        maxpool: 3x5x5
        conv2: 3x3x3
        """
        self.HPARAMS = hparams
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=5,
        )
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.linear1 = nn.Linear(2304, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.HPARAMS["lr"])
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(1))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        diffs = y_hat - y
        return diffs

    def validation_epoch_end(self, outputs):
        diffs = torch.concat(outputs)
        self.log_dict({"mean_abs_diff": torch.mean(torch.abs(diffs)), "std_diff": torch.std(
            diffs)})
