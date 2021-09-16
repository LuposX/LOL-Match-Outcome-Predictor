import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class LaegueDataset_train(Dataset):
    def __init__(self, dataset_location_file, championid_file_location):
        dataset_location = open(open(dataset_location_file).read())
        championidtable_location = open(championid_file_location).read()

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


if __name__ == "__main__":
    traindataset = LaegueDataset_train("../test_dataset_location.txt", "../championidtable_location.txt")
    train_dataloader = DataLoader(traindataset, batch_size=1, shuffle=True)

    print(len(train_dataloader))
