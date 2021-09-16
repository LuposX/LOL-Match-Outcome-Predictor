import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import linecache
import simplejson


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)


class LaegueDataset_train(Dataset):
    def __init__(self, dataset_location_file, championid_file_location):
        # load the dataset
        self._filename = open(dataset_location_file).read()
        self._total_data = 0
        self._total_data = rawgencount(self._filename)

        championidtable_location = open(championid_file_location, encoding='utf8').read()
        championid_to_name = pd.read_csv(championidtable_location)
        self.lookuptable = championid_to_name["key"].values.tolist()

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx_match):
        try:
            line = linecache.getline(self._filename, idx_match + 1).rstrip('\n')
            json_line = simplejson.loads(line)

            # Get the Champion keys of every player in one match
            data = [json_line[i]["championId"] for i in range(10)]

            # convert champion keys into index notation
            data = torch.tensor([self.lookuptable.index(data[i]) for i in range(10)])

            # convert data into one hot vecot encoding
            data = torch.eye(148).index_select(dim=0, index=data)

            target = torch.tensor([int(json_line[1]["stats"]["win"]),
                                   int(json_line[8]["stats"]["win"])
                                   ])

        except Exception as e:
            print("An Exception occurred when trying to load the dataset: " + e)

            line = linecache.getline(self._filename, idx_match + 3).rstrip('\n')
            json_line = simplejson.loads(line)

            # Get the Champion keys of every player in one match
            data = [json_line[i]["championId"] for i in range(10)]

            # convert champion keys into index notation
            data = torch.tensor([self.lookuptable.index(data[i]) for i in range(10)])

            # convert data into one hot vecot encoding
            data = torch.eye(148).index_select(dim=0, index=data)

            target = torch.tensor([int(json_line[1]["stats"]["win"]),
                                   int(json_line[8]["stats"]["win"])
                                   ])

        return data, target


if __name__ == "__main__":
    traindataset = LaegueDataset_train("../test_dataset_location.txt", "../championidtable_location.txt")
    train_dataloader = DataLoader(traindataset, batch_size=2)

    for batch in train_dataloader:
        data, target = batch
        print(data)

