# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import functools
import collections

import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from torchvision import transforms
from torchvision import utils

sys.path.append('./')
from features.custom_features import SPATIAL_FEATURES


class SC2ReplayDataset(Dataset):
    """Dataset for loading StarCraft II replays."""
    def __init__(self,
                 root_dir='./parsed/TvP/',
                 include=['height_map', 'visibility_map', 'player_relative', 'unit_type'],
                 train=True,
                 transform=None,
                 max_timesteps=50):

        self.suffix = 'train' if train else 'test'
        self.root_dir = os.path.join(root_dir, self.suffix)

        self.include = include
        self.transform = transform
        self.max_timesteps = max_timesteps

        npz_pattern = os.path.join(self.root_dir, '**/SpatialFeatures.npz')
        self.npz2length = self.get_replay_lengths(glob.glob(npz_pattern, recursive=True))
        self.npz_files = [f for f, l in self.npz2length.items() if l > self.max_timesteps]
        self.json_files = [
            os.path.join(os.path.dirname(npz_f), 'PlayerMetaInfo.json') for \
                npz_f in self.npz_files
        ]
        self.counts = self.count_class_distribution(self.json_files)
        assert len(self.npz_files) == len(self.json_files)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):

        with np.load(self.npz_files[idx]) as fp:
            name2feature = {k: v for k, v in fp.items()}
        name2feature = {
            k: v for k, v in name2feature.items() if k in self.include
        }

        with open(self.json_files[idx], 'r') as fp:
            meta = json.load(fp)
            if meta['1']['race'] == 'Terran':
                if meta['1']['result'] == 'Victory':
                    terran_wins = 1
                else:
                    terran_wins = 0
            elif meta['1']['race'] == 'Protoss':
                if meta['1']['result'] == 'Defeat':
                    terran_wins = 1
                else:
                    terran_wins = 0

        return name2feature, terran_wins

    def get_replay_lengths(self, npz_files):
        result = {}
        for npz_f in tqdm(npz_files, total=len(npz_files)):
            with np.load(npz_f) as fp:
                replay_len = fp.get('unit_type').__len__()
            result[npz_f] = replay_len

        return result

    def count_class_distribution(self, meta_json_files):

        out = {}

        for json_file in tqdm(meta_json_files, total=len(meta_json_files)):
            with open(json_file, 'r') as f:
                meta_info = json.load(f)

            p1 = '_'.join([meta_info["1"]["race"], meta_info["1"]["result"]])
            p2 = '_'.join([meta_info["2"]["race"], meta_info["2"]["result"]])

            try:
                out[p1] += 1
            except KeyError:
                out[p1] = 1

            try:
                out[p2] += 1
            except KeyError:
                out[p2] = 1

        return out


def replay_collate_fn(samples, max_timesteps=50, weighting='log'):
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

    batched_inputs = collections.defaultdict(list)
    batched_targets = []
    for input_dict, target in samples:

        timesteps, _, _ = input_dict.get('unit_type').shape
        if timesteps < max_timesteps:
            max_timesteps = timesteps

        weights = [weight_fn(i) for i in range(timesteps)]
        weights /= np.sum(weights)
        try:
            timestep_indices = np.random.choice(timesteps, max_timesteps, replace=False, p=weights)
            #timestep_indices = WeightedRandomSampler(weights, max_timesteps, replacement=False)
            timestep_indices = sorted(list(timestep_indices), reverse=False)
        except ValueError as e:
            print(f"Timesteps: {timesteps}")
            print(f"Max timesteps: {max_timesteps}")
            print(f"Weights: {len(weights)}")
            raise ValueError(str(e))

        for name, feature in input_dict.items():
            sampled_feature = feature[timestep_indices]
            if name == 'unit_type':
                mask = sampled_feature > SPATIAL_FEATURES._asdict()['unit_type'].scale + 1
                sampled_feature[mask] = 0
            #if name != 'height_map':
            #    sampled_feature = sampled_feature.astype(np.int32)
            sampled_feature = torch.from_numpy(sampled_feature)
            batched_inputs[name].append(sampled_feature)

        batched_targets.append(target)

    batched_tensor_inputs = {}
    for name, inp in batched_inputs.items():
        batched_tensor_inputs[name] = torch.stack(inp, dim=0)

    out = {
        'inputs': batched_tensor_inputs,
        'targets': torch.FloatTensor(batched_targets),
    }

    return out


if __name__ == '__main__':

    BATCH_SIZE = 2
    WEIGHTING = 'log'
    MAX_TIMESTEPS = 50
    SPATIAL_SPECS = SPATIAL_FEATURES._asdict()
    ROOT = './parsed/TvP/'

    dataset = SC2ReplayDataset(root_dir=ROOT, train=True, max_timesteps=MAX_TIMESTEPS)
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
        inputs_ = batch.get('inputs')     # dictionary of lists
        targets_ = batch.get('targets')   # list

        for j, (name_, feat_) in enumerate(inputs_.items()):
            type_ = str(SPATIAL_SPECS[name_].type).split('.')[-1]
            scale_ = SPATIAL_SPECS[name_].scale
            print(f"[{j:>02}] Name: {name_:<15} | Type: {type_:<11} | Scale: {scale_:>4} | Shape: {feat_.size()}")

        break
