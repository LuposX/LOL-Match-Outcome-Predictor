"""
This File is used for predicting a league of legends match outcome based
on match data like what chamnpions player play.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers


def to_one_hot_vector_encoding(num_classes, input):
    a = torch.tensor([]).cpu()
    for data in input:
        data = data.cpu()
        a = torch.cat((a, torch.eye(num_classes).cpu().index_select(dim=0, index=data).view(-1, 10 * num_classes)))
    return a.cuda()


class NN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            nn.Linear(in_features=148 * 10, out_features=200),  # "*10" because we have 10 players
            nn.PReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.BatchNorm1d(num_features=100),  # Batchnorm only in hidden layers?
            nn.PReLU(),
            nn.Linear(in_features=100, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = to_one_hot_vector_encoding(148, x)
        y_hat = self(x)
        loss = self.loss(y_hat, y.cuda())

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = to_one_hot_vector_encoding(148, x)
        y_hat = self(x)
        loss = self.loss(y_hat, y.cuda())

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def loss(self, input, target):
        return F.binary_cross_entropy(input.float(), target.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    # load model
    pretrained_model = pl.LightningModule.load_from_checkpoint(PATH)
    pretrained_model.freeze()

    # or for prediction
    out = pretrained_model(x)
