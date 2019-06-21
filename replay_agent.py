# -*- coding: utf-8 -*-

"""
    1. ReplayAgent
"""

import os
import collections
import numpy as np

from pysc2.env import environment
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('result_dir', default='./parsed/', help='Directory to write parsed files')


class ReplayAgent(object):
    """A replay agent that saves replay information."""
    def __init__(self, replay_name):

        self.replay_name = replay_name
        self.write_dir = os.path.join(FLAGS.result_dir, replay_name)
        self.screen_features = collections.defaultdict(list)
        self.minimap_features = collections.defaultdict(list)

    def step(self, timestep):
        """..."""

        self._append_screen_features(timestep)
        self._append_minimap_features(timestep)

        if timestep.step_type == environment.StepType.LAST:

            os.makedirs(self.write_dir, exist_ok=True)
            self._save_screen_features(format_='npz')
            self._save_minimap_features(format_='npz')

            return

    def _append_screen_features(self, timestep):
        screen = timestep.observation['feature_screen']
        name2idx = screen._index_names[0]  # name: index
        for name, _ in name2idx.items():
            self.screen_features[name].append(screen[name])

    def _append_minimap_features(self, timestep):
        minimap = timestep.observation['feature_minimap']
        name2idx = minimap._index_names[0]  # name: index
        for name, _ in name2idx.items():
            self.minimap_features[name].append(minimap[name])

    def _save_screen_features(self, format_='npz'):
        """Save screen to .npz format."""
        assert isinstance(self.screen_features, dict)

        if format_ == 'npz':
            np.savez_compressed(
                file=os.path.join(self.write_dir, 'ScreenFeatures.npz'),
                **self.screen_features
            )
        else:
            raise NotImplementedError

    def _save_minimap_features(self, format_='npz'):
        """Save minimap to .npz format."""
        assert isinstance(self.minimap_features, dict)

        if format_ == 'npz':
            np.savez_compressed(
                file=os.path.join(self.write_dir, 'MinimapFeatures.npz'),
                **self.minimap_features
            )
        else:
            raise NotImplementedError
