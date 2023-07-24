import torch
from torch.utils.data import Dataset

from utils.preprocessing import merge_csv_files


class CostumDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data, self.labels = merge_csv_files(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
