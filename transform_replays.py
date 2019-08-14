# -*- coding: utf-8 -*-

"""
    Parsing replays.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import json
import time
import collections
import websocket
import numpy as np

from absl import app
from absl import flags
from absl import logging
from pysc2 import run_configs
from pysc2.env import environment
from pysc2.lib import point, features
from pysc2.lib import protocol
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as sc_common

from parsers import ParserBase
from parsers import ScreenFeatParser
from parsers import MinimapFeatParser
from parsers import SpatialFeatParser

from features.custom_features import custom_features_from_game_info


FLAGS = flags.FLAGS
flags.DEFINE_string('sc2_path', default='C:/Program Files (x86)/StarCraft II/', help='Path to where the client is installed.')
flags.DEFINE_string('replay_file', default=None, help='Path to replay file, optional if replay directory is not specified.')
flags.DEFINE_string('replay_dir', default=None, help='Directory with replays, optional if replay file is not specified.')
flags.DEFINE_string('result_dir', default='./parsed/', help='Directory to write parsed files.')
flags.DEFINE_integer('screen_size', default=64, help='Size of game screen.')
flags.DEFINE_integer('minimap_size', default=64, help='Size of minimap.')
flags.DEFINE_integer('step_mul', default=4, help='Sample interval.')
flags.DEFINE_integer('min_game_length', default=3000, help='Game length lower bound.')
flags.DEFINE_integer('resume_from', default=0, help='Index of replay to resume from.')
flags.DEFINE_float('discount', default=1., help='Not used.')
flags.DEFINE_bool('override', default=False, help='Force overriding existing results.')

flags.DEFINE_enum('race_matchup', default=None, enum_values=['TvT', 'TvP', 'TvZ', 'PvP', 'PvZ', 'ZvZ'], help='Race matchups.')
flags.register_validator('race_matchup', lambda matchup: all([race in ['T', 'P', 'Z'] for race in matchup.split('v')]))

logging.set_verbosity(logging.INFO)


def check_flags():
    """Check validity of command line arguments."""
    if FLAGS.replay_file is not None and FLAGS.replay_dir is not None:
        raise ValueError("Only one of 'replay_file' and 'replay_dir' must be specified.")
    else:
        if FLAGS.replay_file is not None:
            logging.info("Parsing a single replay.")
        elif FLAGS.replay_dir is not None:
            logging.info("Parsing replays in {}".format(FLAGS.replay_dir))
        else:
            raise ValueError("Both 'replay_file' and 'replay_dir' are not specified.")


class ReplayRunner(object):
    """
    Parsing replay data, based on the following implementation:
        https://github.com/narhen/pysc2-replay/blob/master/transform_replay.py
        https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_actions.py
    """
    def __init__(self, replay_file_path, parser_objects,
                 player_id=1, screen_size=(64, 64), minimap_size=(64, 64),
                 discount=1., step_mul=1, override=False):

        self.replay_file_path = os.path.abspath(replay_file_path)
        self.replay_name = os.path.split(replay_file_path)[-1].replace('.SC2Replay', '')
        self.write_dir = os.path.join(FLAGS.result_dir, FLAGS.race_matchup, self.replay_name)

        if isinstance(parser_objects, list):
            self.parsers = [p_obj(self.write_dir) for p_obj in parser_objects]
        elif issubclass(parser_objects, ParserBase):
            self.parsers = [parser_objects(self.write_dir)]
        else:
            raise ValueError("Argument 'parsers' expects a single or list of Parser objects.")

        self.player_id = player_id
        self.discount = discount
        self.step_mul = step_mul
        self.override = override

        # Configure screen size
        if isinstance(screen_size, tuple):
            self.screen_size = screen_size
        elif isinstance(screen_size, int):
            self.screen_size = (screen_size, screen_size)
        else:
            raise ValueError("Argument 'screen_size' requires a tuple of size 2 or a single integer.")

        # Configure minimap size
        if isinstance(minimap_size, tuple):
            self.minimap_size = minimap_size
        elif isinstance(minimap_size, int):
            self.minimap_size = (minimap_size, minimap_size)
        else:
            raise ValueError("Argument 'minimap_size' requires a tuple of size 2 or a single integer.")

        assert len(self.screen_size) == 2
        assert len(self.minimap_size) == 2

        # Arguments for 'sc_process.StarCraftProcess'. Check the following:
        # https://github.com/deepmind/pysc2/blob/master/pysc2/lib/sc_process.py

        try:
            sc2_process_configs = {"full_screen": False, 'timeout_seconds': 300}
            self.run_config = run_configs.get()
            self.sc2_process = self.run_config.start(**sc2_process_configs)
            self.controller = self.sc2_process.controller
        except websocket.WebSocketTimeoutException as e:
            raise ConnectionRefusedError(f'Connection to SC2 process unavailable. ({e})')
        except protocol.ConnectionError as e:
            raise ConnectionRefusedError(f'Connection to SC2 process unavailable. ({e})')

        # Check the following links for usage of run_config and controller.
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/run_configs/platforms.py
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/lib/sc_process.py
        #   https://github.com/deepmind/pysc2/blob/master/pysc2/lib/remote_controller.py

        # Load replay information & check validity.
        replay_data = self.run_config.replay_data(self.replay_file_path)
        info = self.controller.replay_info(replay_data)
        if not self.check_valid_replay(info, self.controller.ping()):
            self.safe_escape()
            raise ValueError('Invalid replay.')

        # Filter replay by race matchup
        if FLAGS.race_matchup is not None:
            if not self.check_valid_matchup(info, matchup=FLAGS.race_matchup):
                self.safe_escape()
                raise ValueError('Invalid matchup.')

        # Map name
        self.map_name = info.map_name

        print('...')
        # 'raw=True' returns enables the use of 'feature_units'
        # https://github.com/Blizzard/s2client-proto/blob/master/docs/protocol.md#interfaces
        interface = sc_pb.InterfaceOptions(
            raw=False,
            score=True,
            show_cloaked=False,
            feature_layer=sc_pb.SpatialCameraSetup(width=24, allow_cheating_layers=True)
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
        self.info = info

    def start(self):
        """Parse replays."""

        if (not self.override) and os.path.isdir(self.write_dir):
            files_to_write = [parser.NPZ_FILE for parser in self.parsers]
            if all([f in os.listdir(self.write_dir) for f in files_to_write]):
                logging.info('This replay has already been parsed.')
                return
        else:
            os.makedirs(self.write_dir, exist_ok=True)

        # Save player meta information (results, apm, mmr, ...)
        player_meta_info = self.get_player_meta_info(self.info)
        with open(os.path.join(self.write_dir, 'PlayerMetaInfo.json'), 'w') as fp:
            json.dump(player_meta_info, fp, indent=4)

        # sc_pb; RequestGameInfo -> ResponseGameInfo
        _features = custom_features_from_game_info(self.controller.game_info())

        while True:

            # Take step, scale specified by 'step_mul' (sc_pb, RequestStep -> ResponseStep)
            self.controller.step(self.step_mul)

            # Receive observation (sc_pb, RequestObservation -> ResponseObservation)
            obs = self.controller.observe()

            # '.transform_obs' is defined under features.Features
            try:
                agent_obs = _features.custom_transform_obs(obs)
            except Exception as err:
                print(err)

            if obs.player_result:
                self._state = environment.StepType.LAST
                discount = 0
            else:
                self._state = environment.StepType.MID
                discount = self.discount

            self._episode_steps += self.step_mul

            step = environment.TimeStep(
                step_type=self._state, reward=0,
                discount=discount, observation=agent_obs
            )

            for parser in self.parsers:
                parser.step(timestep=step)

            if self._state == environment.StepType.LAST:
                break  # break out of while loop

    @staticmethod
    def check_valid_replay(info, ping):
        """Check validity of replay."""
        if info.HasField('error'):
            logging.info('Has error.')
            return False
        elif info.base_build != ping.base_build:
            logging.info('Different base build.')
            return False
        elif info.game_duration_loops < FLAGS.min_game_length:
            logging.info('Game too short.')
            return False
        elif len(info.player_info) != 2:
            logging.info('Not a game with two players.')
            return False
        else:
            return True

    @staticmethod
    def get_player_meta_info(info):
        """Get game results."""
        result = {}
        for info_pb in info.player_info:
            temp_info = {}
            temp_info['race'] = sc_common.Race.Name(info_pb.player_info.race_actual)
            temp_info['result'] = sc_pb.Result.Name(info_pb.player_result.result)
            temp_info['apm'] = info_pb.player_apm
            temp_info['mmr'] = info_pb.player_mmr
            result[info_pb.player_info.player_id] = temp_info

        return result

    @staticmethod
    def check_valid_matchup(info, matchup=None):
        """
        Filter replay by race matchup.
        A typical matchup string takes the form of 'TvT', for example.
        Returns True if matchup is valid, or of interest.
        """

        matchup_list = matchup.split('v')

        full2short = {'Terran': 'T', 'Protoss': 'P', 'Zerg': 'Z'}
        assert all([race in full2short.values() for race in matchup_list])

        races_short = []
        for info_pb in info.player_info:
            race_full = sc_common.Race.Name(info_pb.player_info.race_actual)
            race_short = full2short.get(race_full)
            races_short.append(race_short)

        def compare(x_seq, y_seq):
            is_equal = collections.Counter(x_seq) == collections.Counter(y_seq)
            return is_equal

        if matchup is not None:
            logging.info(
                'Matchup: {}v{}, expected {} (or {}).'.format(*races_short, matchup, matchup[::-1])
            )

        return compare(matchup_list, races_short)

    def safe_escape(self):
        """Closes client."""
        if self.controller.ping():
            self.controller.quit()


def main(argv):
    """Main function."""

    # Check flag sanity
    check_flags()

    # Set path to StarCraft II
    os.environ['SC2PATH'] = FLAGS.sc2_path

    # Get parser objects
    parser_objects = [ScreenFeatParser, MinimapFeatParser, SpatialFeatParser]

    def _main(replay_file, parser_objs):
        try:
            runner = ReplayRunner(
                replay_file_path=replay_file,
                parser_objects=parser_objs,
                screen_size=FLAGS.screen_size,
                minimap_size=FLAGS.minimap_size,
                discount=FLAGS.discount,
                step_mul=FLAGS.step_mul,
                override=FLAGS.override
            )
            runner.start()
        except ValueError as e:
            logging.info(str(e))
        except ConnectionRefusedError as e:
            logging.info(str(e))
        except KeyboardInterrupt:
            sys.exit()
        finally:
            try:
                runner.safe_escape()
            except UnboundLocalError:
                pass

    if FLAGS.replay_file is not None:
        _main(FLAGS.replay_file, parser_objects)
    elif FLAGS.replay_dir is not None:
        replay_files = glob.glob(os.path.join(FLAGS.replay_dir, '/**/*.SC2Replay'), recursive=True)
        replay_files = replay_files[FLAGS.resume_from:]
        for i, replay_file in enumerate(replay_files):
            if i < FLAGS.resume_from:
                continue
            logging.info(
                'Path to replay:\n{}'.format(replay_file.split()[-1])
            )
            _main(replay_file, parser_objects)
            logging.info(
                'Replay [{:>05d}/{:>05d}] terminating.'.format(i + 1, len(replay_files))
            )
            time.sleep(3.)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    app.run(main)
