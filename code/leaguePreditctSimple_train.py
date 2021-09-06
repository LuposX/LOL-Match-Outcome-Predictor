"""
Author: Lupos
Date: 20.05.20
Purpose: Predict the outcome of a League Match with a Neural Network. Train the Neural Network for that.
"""

import comet_ml
import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from torch.utils.data import DataLoader, Dataset

import pandas as pd

import os
from argparse import Namespace
from pathlib import Path
import shutil
import json


def to_one_hot_vector_encoding(num_classes, input):
    a = torch.tensor([]).cpu()
    for data in input:
        data = data.cpu()
        a = torch.cat((a, torch.eye(num_classes).cpu().index_select(dim=0, index=data).view(-1, 10 * num_classes)))
    return a



class LaegueDataset_train(Dataset):
    def __init__(self):
        dataset_location = open(open("../dataset_location.txt").read())
        championidtable_location = open("../championidtable_location.txt").read()

        self.dataset_train = json.load(dataset_location)

        self.championid_to_name = pd.read_csv(championidtable_location)

        self.lookuptable = self.championid_to_name["key"].values.tolist()

    def __len__(self):
        return len(self.dataset_train["data"])

    def __getitem__(self, idx_match):
        try:

            data = [self.dataset_train["data"][idx_match][i]["championId"] for i in
                    range(10)]  # Get the Champion keys of every player in one match
            data = torch.tensor(
                [self.lookuptable.index(data[i]) for i in range(10)])  # convert champion keys into index notation

            # convert data into one hot vecot encoding
            data = torch.eye(148).index_select(dim=0, index=data)

            # convert it into one hot vector encoding
            data = torch.eye(148).index_select(dim=0, index=data)
            data = data.flatten()

            target = torch.tensor([int(self.dataset_train["data"][idx_match][1]["stats"]["win"]),
                                   int(self.dataset_train["data"][idx_match][8]["stats"]["win"])
                                   ])

        except Exception as e:
            print(f"An Exception occurred when trying to load the dataset: {e}")

            data = [self.dataset_train["data"][idx_match + 1][i]["championId"] for i in
                    range(10)]  # Get the Champion keys of every player in one match
            data = torch.tensor([self.lookuptable.index(data[i]) for i in range(10)])  # convert champion keys into ids

            # convert it into one hot vector encoding
            data = torch.eye(148).index_select(dim=0, index=data)
            data = data.flatten()

            target = torch.tensor([int(self.dataset_train["data"][idx_match + 1][1]["stats"]["win"]),
                                   int(self.dataset_train["data"][idx_match + 1][8]["stats"]["win"])
                                   ])

        return data, target


class NN(pl.LightningModule):
    def __init__(self, hparams, experiment_name):
        super().__init__()

        # self.hparams = hparams  # Depreceated in newer versions
        self.save_hyperparameters(hparams)

        self.checkpoint_folder = "LeaguePredictCheckpoints/"

        self.experiment_name = experiment_name

        # creating checkpoint folder
        dirpath = Path(self.checkpoint_folder)
        if not dirpath.exists():
            os.makedirs(dirpath, 0o755)

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=1480, out_features=50),  # "*10" because we have 10 players
            nn.PReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=50, out_features=20),
            nn.BatchNorm1d(num_features=20),  # Batchnorm only in hidden layers?
            nn.PReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(in_features=20, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return {"loss": loss, "log": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return {"val_loss": loss, "log": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def loss(self, input, target):
        return F.binary_cross_entropy(input.float(), target.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.LR)

    def train_dataloader(self):
        train_dataset = LaegueDataset_train()
        return DataLoader(train_dataset, num_workers=self.hparams.NUMWORK, batch_size=self.hparams.BATCHSIZE)

    def val_dataloader(self):
        val_dataset = LaegueDataset_train()
        return DataLoader(val_dataset, num_workers=self.hparams.NUMWORK, batch_size=self.hparams.BATCHSIZE)

    def on_epoch_end(self) -> None:
        if self.current_epoch % self.hparams.SAVE_MODEL_EVERY_EPOCH == 0:
            trainer.save_checkpoint(
                self.checkpoint_folder + "/" + self.experiment_name + "_epoch_" + str(self.current_epoch) + ".ckpt")
            comet_logger.experiment.log_asset_folder(self.checkpoint_folder, step=self.current_epoch)

            # Deleting the folder where we saved the model so that we dont upload a thing twice
            dirpath = Path(self.checkpoint_folder)
            if dirpath.exists() and dirpath.is_dir():
                shutil.rmtree(dirpath)

            # creating checkpoint folder
            access_rights = 0o755
            os.makedirs(dirpath, access_rights)

    def on_train_end(self):
        trainer.save_checkpoint(
            self.checkpoint_folder + "/" + self.experiment_name + "_epoch_" + str(self.current_epoch) + ".ckpt")
        comet_logger.experiment.log_asset_folder(self.checkpoint_folder, step=self.current_epoch)


if __name__ == "__main__":
    # Parameter for the NN
    args = {
        "EPOCHS": 2,
        "LR": 0.003,
        "AUTO_LR": False,
        # Runs a learning rate finder algorithm(https://arxiv.org/abs/1506.01186) before any training, to find optimal initial learning rate.
        "BENCHMARK": True,  # This flag is likely to increase the speed of your system if your input sizes don’t change.
        "NUMWORK": 1,
        "BATCHSIZE": 1,  # needs to be smaller than the size of dataset
        "SAVE_MODEL_EVERY_EPOCH": 1,
    }
    hparams = Namespace(**args)

    # Parameters
    experiment_name = "leaguePredict"
    dataset_name = "league_ranked_2020_from_kaggle"
    checkpoint_folder = "LeaguePredictCheckpoints/"
    dirpath = Path(checkpoint_folder)

    # init logger
    comet_logger = loggers.CometLogger(
        api_key="hm3iPCrv0LDXFj8AHNO0uhBP4",
        rest_api_key="Z572EjCDYjwGCoejWL817C0rW",
        project_name="LeaguePredict",
        experiment_name=experiment_name,
        # experiment_key="222a685177474cb9b358b5ee642564dc"  # used for resuming trained id can be found in comet.ml
    )

    # Init the Neural Network(NN)
    net = NN(hparams, experiment_name)

    # logging
    comet_logger.experiment.set_model_graph(str(net))
    comet_logger.experiment.add_tags(tags=["testing"])
    comet_logger.experiment.log_dataset_info(dataset_name)

    # deleting the checkpoint folder
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    # Init our trainer
    trainer = Trainer(gpus=0,
                      max_epochs=args["EPOCHS"],
                      logger=comet_logger,
                      # auto_lr_find=args["AUTO_LR"],
                      benchmark=args["BENCHMARK"],
                      weights_summary="full"
                      )

    # Start training the NN
    trainer.fit(net)
