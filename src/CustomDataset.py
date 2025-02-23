import random
from abc import ABC
import torch
from torch.utils.data import Dataset
import config
import os
from glob import glob
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

random.seed(42)

class CustomRawDataset(Dataset, ABC):
    def __init__(self, type_of_data = "train"):
        if type_of_data not in ["train", "val", "test"]:
            raise Exception("Invalid value passed for type_of_data, Valid values are 'train', 'val', 'test'")
        self.data_dir = os.path.join(config.INPUT_FINAL, type_of_data)
        self.list_of_files = glob(os.path.join(self.data_dir, "*", "*.pt"))
        random.shuffle(self.list_of_files)

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        file_name = self.list_of_files[index]
        gesture = os.path.basename(os.path.dirname(file_name))
        X = torch.load(file_name)
        return X, config.CLASS_DICT[gesture]

def custom_collate_fn(batch):
    Xs, Ys = zip(*batch)
    X_lens = [len(x) for x in Xs]
    X_zipped = list(zip(Xs, X_lens, Ys))
    Xs, X_lens, Ys = list(zip(*sorted(X_zipped, key=lambda x : x[1], reverse=True)))
    Xs = [elem for elem in Xs]
    Ys = torch.from_numpy(np.array([elem for elem in Ys]))
    Xs_pad = pad_sequence(sequences=Xs, batch_first=True,padding_value=0)
    return Xs_pad, Ys.long(), X_lens