# -*- coding: utf-8 -*-

"""
    Parsing replays.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import json
import importlib

from absl import app, flags
from pysc2 import run_configs
from pysc2.env import environment
from pysc2.lib import point, features
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as sc_common

FLAGS = flags.FLAGS
flags.DEFINE_string('sc2_path', default='C:/Program Files (x86)/StarCraft II/', help='Path to where the client is installed.')
flags.DEFINE_string('replay_file', default=None, help='Path to replay file, optional if replay directory is not specified.')
flags.DEFINE_string('replay_dir', default=None, help='Directory with replays, optional if replay file is not specified.')
# flags.DEFINE_string('result_dir', default='./parsed/', help='Directory to write parsed files')  # Defined in replay_agent.py
flags.DEFINE_string('agent', default='replay_agent.ReplayAgent', help='path to agent class, relative.')
flags.DEFINE_integer('screen_size', default=64, help='Size of game screen.')
flags.DEFINE_integer('minimap_size', default=64, help='Size of minimap.')
flags.DEFINE_integer('step_mul', default=4, help='Sample interval.')
flags.DEFINE_integer('min_game_length', default=3000, help='Game length lower bound.')
flags.DEFINE_float('discount', default=1., help='Not used.')


def check_flags():
    """Check validity of command line arguments."""
    if FLAGS.replay_file is not None and FLAGS.replay_dir is not None:
        raise ValueError("Only one of 'replay_file' and 'replay_dir' must be specified.")
    else:
        if FLAGS.replay_file is not None:
            print("Parsing a single replay.")
        elif FLAGS.replay_dir is not None:
            print("Parsing replays in {}".format(FLAGS.replay_dir))
        else:
            raise ValueError("Both 'replay_file' and 'replay_dir' not specified.")

    if FLAGS.screen_size != FLAGS.minimap_size:
        raise ValueError("Only supports equal values for 'screen_size' and 'minimap_size'.")


class ReplayParser(object):
    """
    Parsing replay data, based on the following implementation:
        https://github.com/narhen/pysc2-replay/blob/master/transform_replay.py
        https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_actions.py
    """
    def __init__(self, replay_file_path, agent,
                 player_id=1, screen_size=(64, 64), minimap_size=(64, 64),
                 discount=1., step_mul=1):

        self.agent = agent
        self.discount = discount
        self.step_mul = step_mul
        self.player_id = player_id
        
        self.replay_file_path = os.path.abspath(replay_file_path)
        self.replay_name = os.path.split(replay_file_path)[-1].replace('.SC2Replay', '')

        # Configure screen size
        if isinstance(screen_size, tuple):
            self.screen_size = screen_size
        elif isinstance(screen_size, int):
            self.screen_size = (screen_size, screen_size)
        else:
            raise ValueError

        # Configure minimap size
        if isinstance(minimap_size, tuple):
            self.minimap_size = minimap_size
        elif isinstance(minimap_size, int):
            self.minimap_size = (minimap_size, minimap_size)
        else:
            raise ValueError

        assert len(self.screen_size) == 2
        assert len(self.minimap_size) == 2

        self.run_config = run_configs.get()
        self.sc2_process = self.run_config.start()
        self.controller = self.sc2_process.controller

        # Check the following links for usage of run_config and controller.
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/run_configs/platforms.py
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/lib/sc_process.py
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/lib/remote_controller.py

        replay_data = self.run_config.replay_data(self.replay_file_path)  # Read replay file
        ping = self.controller.ping()

        # 'replay_info' returns metadata about a replay file. Does not load the replay
        info = self.controller.replay_info(replay_data)
        if not self.check_valid_replay(info, ping):
            print("'{}.SC2Replay' is not a valid replay file.".format(self.replay_name))
            raise ValueError

        # Map name
        self.map_name = info.map_name

        # Save player meta information (TODO: Add more info)
        self.player_meta_info = {}
        for info_pb in info.player_info:
            temp_info = {}
            temp_info['race'] = sc_common.Race.Name(info_pb.player_info.race_actual)
            temp_info['result'] = sc_pb.Result.Name(info_pb.player_result.result)
            temp_info['apm'] = info_pb.player_apm
            temp_info['mmr'] = info_pb.player_mmr
            self.player_meta_info[info_pb.player_info.player_id] = temp_info

        self.write_dir = os.path.join(FLAGS.result_dir, self.replay_name)
        os.makedirs(self.write_dir, exist_ok=True)
        with open(os.path.join(self.write_dir, 'PlayerMetaInfo.json'), 'w') as fp:
            json.dump(self.player_meta_info, fp, indent=4)

        interface = sc_pb.InterfaceOptions(
            raw=False,
            score=True,
            show_cloaked=False,
            feature_layer=sc_pb.SpatialCameraSetup(width=24)
        )

        self.screen_size = point.Point(*self.screen_size)
        self.minimap_size = point.Point(*self.minimap_size)
        self.screen_size.assign_to(interface.feature_layer.resolution)
        self.minimap_size.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if info.local_map_path:
            map_data = self.run_config.map_data(info.local_map_path)

        self._episode_length = info.game_duration_loops
        self._episode_steps = 0

        # Request replay
        self.controller.start_replay(
            req_start_replay=sc_pb.RequestStartReplay(
                replay_data=replay_data,
                map_data=map_data,
                options=interface,
                observed_player_id=self.player_id,
                disable_fog=True,
            )
        )

        self._state = environment.StepType.FIRST

    def start(self):
        """Start parsing replays."""
        # 'game_info()' returns static data about the current game and map
        _features = features.features_from_game_info(self.controller.game_info())

        while True:

            # Take step, scale specified by 'step_mul'
            self.controller.step(self.step_mul)

            # Receive observation
            obs = self.controller.observe()
            try:
                # 'transform_obs' is defined under features.Features
                agent_obs = _features.transform_obs(obs)
            except Exception as e:
                print(str(e), '...')

            if obs.player_result:
                self._state = environment.StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += self.step_mul

            step = environment.TimeStep(
                step_type=self._state, reward=0,
                discount=discount, observation=agent_obs
            )
            self.agent.step(step)

            if obs.player_result:
                break

            self._state = environment.StepType.MID

    @staticmethod
    def check_valid_replay(info, ping):
        """Add function docstring."""
        if info.HasField('error'):
            print('Has error...')
            return False
        elif info.base_build != ping.base_build:
            print('Different base build...')
            return True
        elif info.game_duration_loops < FLAGS.min_game_length:
            print('Game too short...')
            return False
        elif len(info.player_info) != 2:
            print('Not a game with two players...')
            return False
        else:
            return True


def main(unused):
    """Main function."""

    # Check flag sanity
    check_flags()

    # Set path to StarCraft II
    os.environ['SC2PATH'] = FLAGS.sc2_path

    # Get agent object
    agent_module, agent_name = FLAGS.agent.rsplit(".", maxsplit=1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    # Parse
    if FLAGS.replay_file is not None:
        replay_name = os.path.split(FLAGS.replay_file)[-1].replace('.SC2Replay', '')
        try:
            parser = ReplayParser(
                replay_file_path=FLAGS.replay_file,
                agent=agent_cls(replay_name=replay_name),
                screen_size=FLAGS.screen_size,
                minimap_size=FLAGS.minimap_size,
                discount=FLAGS.discount,
                step_mul=FLAGS.step_mul
            )
            parser.start()
        except ValueError:
            print("Skipping invalid replay...")
        finally:
            print('Replay finished...')
    else:
        replay_files = glob.glob(os.path.join(FLAGS.replay_dir, '/**/*.SC2Replay'), recursive=True)
        num_replays = len(replay_files)
        print("Parsing {} replays.".format(num_replays))
        for i, replay_file in enumerate(replay_files):
            replay_name = os.path.split(replay_file)[-1].replace('.SC2Replay', '')
            try:
                parser = ReplayParser(
                    replay_file_path=replay_file,
                    agent=agent_cls(replay_name=replay_name),
                    screen_size=FLAGS.screen_size,
                    minimap_size=FLAGS.minimap_size,
                    discount=FLAGS.discount,
                    step_mul=FLAGS.step_mul
                )
                parser.start()
            except ValueError:
                print("Skipping invalid replay...")
            finally:
                print('Replay #{:{align}{width}} finished...'.format(i + 1, align='>', width=num_replays))


if __name__ == '__main__':
    app.run(main)
