# -*- coding: utf-8 -*-

"""
    1. ParserBase
    2. ScreenFeatParser
    3. MinimapFeatParser
    4. CustomSpatialParser
"""

import os
import collections
import numpy as np

from pysc2.env import environment
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('result_dir', default='./parsed/', help='Directory to write parsed files')


class ParserBase(object):
    """Abstract class for replay parsers."""
    def __init__(self, replay_name):
        self.replay_name = replay_name
        self.write_dir = os.path.join(FLAGS.result_dir, replay_name)
        os.makedirs(self.write_dir, exist_ok=True)

    def step(self, timestep):
        """..."""
        if timestep.step_type == environment.StepType.LAST:
            self.save()
            return
        else:
            self.parse(timestep)

    def parse(self, timestep):
        """Must override."""
        raise NotImplementedError

    def save(self, save_format='npz'):
        """Must override."""
        raise NotImplementedError


class ScreenFeatParser(ParserBase):
    """Parse 'feature_screen' from timestep observation."""
    def __init__(self, replay_name):
        super(ScreenFeatParser, self).__init__(replay_name)
        self.screen_features = collections.defaultdict(list)

    def parse(self, timestep):
        self._append_screen_features(timestep)

    def save(self, save_format='npz'):
        self._save_screen_features(format_=save_format)

    def _append_screen_features(self, timestep):
        screen = timestep.observation['feature_screen']
        name2idx = screen._index_names[0]  # name: index
        for name, _ in name2idx.items():
            self.screen_features[name].append(screen[name])

    def _save_screen_features(self, format_):
        """Save screen to .npz format."""
        assert isinstance(self.screen_features, dict)

        if format_ == 'npz':
            write_file = os.path.join(self.write_dir, 'ScreenFeatures.npz')
            np.savez_compressed(
                file=write_file,
                **self.screen_features
            )
            print('{} | Saved screen features to: {}'.format(self.__class__.__name__, write_file))
        else:
            raise NotImplementedError


class MinimapFeatParser(ParserBase):
    """Parse 'feature_minimap' from timestep observation."""
    def __init__(self, replay_name):
        super(MinimapFeatParser, self).__init__(replay_name)
        self.minimap_features = collections.defaultdict(list)

    def parse(self, timestep):
        self._append_minimap_features(timestep)

    def save(self, save_format='npz'):
        self._save_minimap_features(format_=save_format)

    def _append_minimap_features(self, timestep):
        minimap = timestep.observation['feature_minimap']
        name2idx = minimap._index_names[0]  # name: index
        for name, _ in name2idx.items():
            self.minimap_features[name].append(minimap[name])

    def _save_minimap_features(self, format_):
        """Save minimap to .npz format."""
        assert isinstance(self.minimap_features, dict)

        if format_ == 'npz':
            write_file = os.path.join(self.write_dir, 'MinimapFeatures.npz')
            np.savez_compressed(
                file=write_file,
                **self.minimap_features
            )
            print('{} | Saved minimap features to: {}'.format(self.__class__.__name__, write_file))
        else:
            raise NotImplementedError


class SpatialFeatParser(ParserBase):
    """
    Parse 'feature_spatial' from timestep observation.
    Note that 'feature_spatial' is a customly implemented feature.
    """
    def __init__(self, replay_name):
        super(SpatialFeatParser, self).__init__(replay_name)
        self.spatial_features = collections.defaultdict(list)

    def parse(self, timestep):
        self._append_spatial_features(timestep)

    def save(self, save_format='npz'):
        self._save_spatial_features(format_=save_format)

    def _append_spatial_features(self, timestep):
        spatial = timestep.observation['feature_spatial']
        name2idx = spatial._index_names[0]  # name: index
        for name, _ in name2idx.items():
            self.spatial_features[name].append(spatial[name])

    def _save_spatial_features(self, format_):
        """Save 'spatial' to .npz format."""
        assert isinstance(self.spatial_features, dict)

        if format_ == 'npz':
            write_file = os.path.join(self.write_dir, 'SpatialFeatures.npz')
            np.savez_compressed(
                file=write_file,
                **self.spatial_features
            )
            print('{} | Saved Spatial features to: {}'.format(self.__class__.__name__, write_file))
        else:
            raise NotImplementedError
