# -*- coding: utf-8 -*-

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append('./')
from custom_features import SPATIAL_FEATURES


SPATIAL_SPECS = SPATIAL_FEATURES._asdict()


class SC2ReplayDataset(Dataset):
    """Add class doctstring."""
    def __init__(self, root_dir='./parsed/TvT/', transform=None):

        self.root_dir = root_dir  # example: /parsed/TvT/
        self.transform = transform

        npz_pattern = os.path.join(self.root_dir, '**/SpatialFeatures.npz')
        self.npz_files = glob.glob(npz_pattern, recursive=True)
        self.json_files = [
            os.path.join(os.path.dirname(npz_f), 'PlayerMetaInfo.json') for \
                npz_f in self.npz_files
            ]
        assert len(self.npz_files) == len(self.json_files)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        with np.load(self.npz_files[idx]) as fp:
            name2feature = {k: v for k, v in fp.items()}

        with open(self.json_files[idx], 'r') as fp:
            meta = json.load(fp)
            player_one_wins = 1 if meta['1']['result'] == 'Victory' else 0  # FIXME

        return name2feature, player_one_wins


def replay_collate_fn(batch):
    return batch

if __name__ == '__main__':

    BATCH_SIZE = 2

    dataset = SC2ReplayDataset(root_dir='./parsed/TvT/')
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=replay_collate_fn,  # FIXME
    )

    print(f'Number of batches per epoch: {len(dataloader)}')
    print("[Spatial features (Customized)]")
    print('=' * 90)

    batch = next(iter(dataloader))

    for i, batch in enumerate(dataloader):
        
        for sample in batch:
        
            for  j, (name, feat) in enumerate(sample[0].items()):

                type_ = str(SPATIAL_SPECS[name].type).split('-')[-1]
                scale_ = SPATIAL_SPECS[name].scale
                print(f"[{j:>02}] Name: {name:<15} | Type: {type_:<11} | Scale: {scale_:>4} | Shape: {feat.shape}")

            break
        break
    