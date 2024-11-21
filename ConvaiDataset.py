import h5py
import torch
from torch.utils.data import Dataset

class ConvaiDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.h5_data = h5py.File(h5_file, 'r')
        self.keys = list(self.h5_data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        audio = torch.tensor(self.h5_data[key]['audio'][:])
        text_bytes = self.h5_data[key]['text'][()]
        text = text_bytes.decode('utf-8')
        return audio, text