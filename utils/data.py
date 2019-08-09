# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import functools
import collections

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from torchvision import transforms
from torchvision import utils

sys.path.append('./')
from custom_features import SPATIAL_FEATURES


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
            player_one_wins = 1 if meta['1']['result'] == 'Victory' else 0
            player_two_wins = 1 if meta['2']['result'] == 'Victory' else 0

            if player_one_wins == player_two_wins:
                raise NotImplementedError('Draws are not allowed.')

        return name2feature, player_one_wins


def replay_collate_fn(samples, max_timesteps=100, weighting='log'):
    """
    Arguments:
        samples: a list of tuples (x: dict, y: int)
        max_timesteps: The maximum number of timesteps to allow.
            Frames will be sampled with normalized exponential weights against time.
        weighting: str, one of 'exp' or 'log'.
    """

    if weighting == 'exp':
        weight_fn = np.exp
    elif weighting == 'log':
        weight_fn = lambda x: np.log(1 + x)
    else:
        raise NotImplementedError

    out = collections.defaultdict(list)
    for inputs, target in samples:

        timesteps, _, _ = inputs.get('unit_type').shape
        if timesteps < max_timesteps:
            max_timesteps = timesteps

        weights = [weight_fn(i) for i in range(timesteps)]
        weights /= np.sum(weights)
        timestep_indices = WeightedRandomSampler(weights, max_timesteps, replacement=False)
        timestep_indices = sorted(timestep_indices, reverse=False)

        inputs_new = {}
        for name, feat in inputs.items():
            sampled_feat = feat[timestep_indices]
            inputs_new[name] = sampled_feat

        out['inputs'].append(inputs_new)
        out['target'].append(target)

    return out


if __name__ == '__main__':

    BATCH_SIZE = 2
    WEIGHTING = 'log'
    MAX_TIMESTEPS = 50
    SPATIAL_SPECS = SPATIAL_FEATURES._asdict()
    
    dataset = SC2ReplayDataset(root_dir='./parsed/TvT/')
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=functools.partial(
            replay_collate_fn,
            max_timesteps=MAX_TIMESTEPS,
            weighting=WEIGHTING
        ),
    )

    print(f"Number of batches per epoch: {len(dataloader)}")
    print("[Spatial features (Customized)]\n", "=" * 80)

    for i, batch in enumerate(dataloader):

        assert isinstance(batch, dict)
        print(f"Batch size: {len(batch)}")

        inputs_ = batch.get('inputs')    # list
        targets_ = batch.get('target')  # list

        for x, y in zip(inputs_, targets_):
            for  j, (name_, feat_) in enumerate(x.items()):

                type_ = str(SPATIAL_SPECS[name_].type).split('-')[-1]
                scale_ = SPATIAL_SPECS[name_].scale
                print(f"[{j:>02}] Name: {name_:<15} | Type: {type_:<11} | Scale: {scale_:>4} | Shape: {feat_.shape}")

            break
        break
