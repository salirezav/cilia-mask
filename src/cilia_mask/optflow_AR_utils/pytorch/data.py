import glob
import os.path
from pathlib import Path

import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class CiliaData(pl.LightningDataModule):
    def __init__(self, batch_size = 16, flow_type = "nvidia"):
        super().__init__()
        self.root = '/space/cilia/scipy23'
        self.batch_size = batch_size
        self.flow_type = flow_type
        self.cpus = os.cpu_count()
        self.prepare_data_per_node = False
    
    # Why does this function exist? Why not remove it entirely?
    def prepare_data(self):
        pass

    # Why does this function exist? Why not do this in __init__?
    def _setup(self, val_percent = 0.1):
        dataset = CiliaDataset(
            self.root, self.flow_type, is_train = True
        )
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        self.train_data, self.val_data = random_split(dataset, [n_train, n_val])
        self.test_data = CiliaDataset(
            self.root, self.flow_type, is_train = False
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size = self.batch_size, shuffle = True, num_workers = self.cpus
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size = self.batch_size, shuffle = False, num_workers = self.cpus
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size = self.batch_size, shuffle = False, num_workers = self.cpus
        )

class CiliaDataset(Dataset):
    def __init__(self, root, flow_type, is_train = True):
        self.root = root            # Root data directory.
        self.flow_type = flow_type  # Currently: "nvidia" or "opencv"
        
        # Pull all the data points together.
        self.data = glob.glob(os.path.join(self.root, "*", "images", "*.npy"))
        

    def __getitem__(self, index):
        # Some prefixing.
        item = self.data[index]
        item_prefix = item.split("/")[-1].split(".")[0]
        prefix = Path(item).parent.parent.name # ??!?!??
        data_path = os.path.join(self.root, prefix)
        num = item.split("_")[-1].split(".")[0]

        # Load the image.
        image = np.load(item)

        # Now the mask.
        mask_path = os.path.join(data_path, "masks", self.flow_type, f"{prefix}_f_{num}.npy")
        mask_arr = np.array(np.load(mask_path), dtype = np.float32)

        # Prep the image and mask.
        mask = torch.from_numpy(np.expand_dims(mask_arr, axis = 0))
        img = np.array(image, dtype = np.float32)
        img -= img.min()
        img /= img.max()
        img = torch.from_numpy(np.expand_dims(img, axis = 0))

        # Return both, and the id
        return {"image": img, "mask": mask, "id": item_prefix}

    def __len__(self):
        return len(self.data)