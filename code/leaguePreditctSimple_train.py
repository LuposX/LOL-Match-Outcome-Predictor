"""
Author: Lupos
Date: 20.05.20
Purpose: Predict the outcome of a League Match with a Neural Network. Train the Neural Network for that.
"""

# It varies wth "lreaguePredict" that this doesnt use comet

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
    return a.cuda()


class LaegueDataset_train(Dataset):
    def __init__(self, dataset_location, championidtable_location):
        dataset = json.load(dataset_location)
        self.dataset_train = dataset
        self.dataset_train.columns = ["random", "participants"]
        del dataset

        self.championid_to_name = pd.read_csv(
            "E:/Datasets/league_of_legends_ranked_games_2020/champion_and_items/riot_champion.csv")

        self.lookuptable = self.championid_to_name["key"].values.tolist()

    def __len__(self):
        return len(self.dataset_train)

    def __getitem__(self, idx_match):
        try:
            data = [self.dataset_train["data"][idx_match][i]["championId"] for i in
                    range(10)]  # Get the Champion keys of every player in one match
            data = torch.tensor([self.lookuptable.index(data[i]) for i in range(10)])  # convert champion keys into ids

            target = torch.tensor([int(self.dataset_train["data"][idx_match][1]["stats"]["win"]),
                                   int(self.dataset_train["data"][idx_match][8]["stats"]["win"])
                                   ])

        except Exception as e:
            print("An Exception occurred when trying to load the dataset: " + e)

            data = [self.dataset_train["data"][idx_match + 1][i]["championId"] for i in
                    range(10)]  # Get the Champion keys of every player in one match
            data = torch.tensor([self.lookuptable.index(data[i]) for i in range(10)])  # convert champion keys into ids

            target = torch.tensor([int(self.dataset_train["data"][idx_match + 1][1]["stats"]["win"]),
                                   int(self.dataset_train["data"][idx_match + 1][8]["stats"]["win"])
                                   ])

        return data, target


class NN(pl.LightningModule):
    def __init__(self, hparams, dataset_location, championidtable_location):
        super().__init__()

        self.hparams = hparams
        self.checkpoint_folder = "LeaguePredictCheckpoints/"
        self.dataset_location = dataset_location
        self.championidtable_location = championidtable_location

        # creating checkpoint folder
        dirpath = Path(self.checkpoint_folder)
        if not dirpath.exists():
            os.makedirs(dirpath, 0o755)

        self.main = nn.Sequential(
            nn.Linear(in_features=148 * 10, out_features=50),  # "*10" because we have 10 players
            nn.PReLU(),
            nn.Linear(in_features=50, out_features=20),
            nn.BatchNorm1d(num_features=20),  # Batchnorm only in hidden layers?
            nn.PReLU(),
            nn.Linear(in_features=20, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = to_one_hot_vector_encoding(148, x)
        y_hat = self(x)
        loss = self.loss(y_hat, y.cuda())

        return {"loss": loss, "log": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = to_one_hot_vector_encoding(148, x)
        y_hat = self(x)
        loss = self.loss(y_hat, y.cuda())

        return {"val_loss": loss, "log": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def loss(self, input, target):
        return F.binary_cross_entropy(input.float(), target.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        train_dataset = LaegueDataset_train(self.dataset_location,  self.championidtable_location)
        return DataLoader(train_dataset, num_workers=hparams.NUMWORK, batch_size=hparams.BATCHSIZE)

    def val_dataloader(self):
        val_dataset = LaegueDataset_train(self.dataset_location,  self.championidtable_location)
        return DataLoader(val_dataset, num_workers=hparams.NUMWORK, batch_size=hparams.BATCHSIZE)

    def on_epoch_end(self) -> None:
        if self.current_epoch % self.hparams.save_model_every_epoch == 0:
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
        "lr": 0.003,
        "AUTO_LR": False,
        # Runs a learning rate finder algorithm(https://arxiv.org/abs/1506.01186) before any training, to find optimal initial learning rate.
        "BENCHMARK": True,  # This flag is likely to increase the speed of your system if your input sizes don’t change.
        "NUMWORK": 1,
        "BATCHSIZE": 100,
        "save_model_every_epoch": 1
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
    net = NN(hparams, "E:/Datasets/league_of_legends_ranked_games_2020/champion_and_items/league_ranked_2020_short_for_testing.json",
             "E:/Datasets/league_of_legends_ranked_games_2020/champion_and_items/riot_champion.csv")

    # logging
    comet_logger.experiment.set_model_graph(str(net))
    comet_logger.experiment.add_tags(tags=["testing"])
    comet_logger.experiment.log_dataset_info(dataset_name)

    # deleting the checkpoint folder
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    # Init our trainer
    trainer = Trainer(gpus=1,
                      max_epochs=args["EPOCHS"],
                      logger=comet_logger,
                      # auto_lr_find=args["AUTO_LR"],
                      benchmark=args["BENCHMARK"]
                      )

    # Start training the NN
    trainer.fit(net)
