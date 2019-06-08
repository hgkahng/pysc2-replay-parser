# -*- coding: utf-8 -*-

"""
    1. ReplayAgent
"""

class ReplayAgent(object):
    """A replay agent to parse replay information."""
    def step(self, timestep, actions):
        """Step function."""
        print(
            '{}'.format(timestep.observation['game_loop']),
        )
