# -*- coding: utf-8 -*-

"""
    Customized Spatial Features.
"""

import collections
import six
import numpy as np

from pysc2.lib import features
from pysc2.lib import stopwatch
from pysc2.lib import static_data
from pysc2.lib import named_array


sw = stopwatch.sw


class SpatialFeatures(collections.namedtuple("SpatialFeatures", [
        "height_map", "visibility_map", "creep", "camera",
        "player_id", "player_relative", "selected", "unit_type"])):
    """
    Set of customized feature layers (currently 8).
    Similar to pysc2.lib.features.MinimapFeatures, but with 'unit_type'.
    """
    __slots__ = ()

    def __new__(cls, **kwargs):
        feats = {}
        for name, (scale, type_, palette) in six.iteritems(kwargs):
            feats[name] = features.Feature(
                index=SpatialFeatures._fields.index(name),
                name=name,
                layer_set="minimap_renders",
                full_name="spatial " + name,
                scale=scale,
                type=type_,
                palette=palette(scale) if callable(palette) else palette,
                clip=False,
            )

        return super(SpatialFeatures, cls).__new__(cls, **feats)  # pytype: disable=missing-parameter


SPATIAL_FEATURES = SpatialFeatures(
    height_map=(256, features.FeatureType.SCALAR, features.colors.winter),
    visibility_map=(4, features.FeatureType.CATEGORICAL, features.colors.VISIBILITY_PALETTE),
    creep=(2, features.FeatureType.CATEGORICAL, features.colors.CREEP_PALETTE),
    camera=(2, features.FeatureType.CATEGORICAL, features.colors.CAMERA_PALETTE),
    player_id=(17, features.FeatureType.CATEGORICAL, features.colors.PLAYER_ABSOLUTE_PALETTE),
    player_relative=(5, features.FeatureType.CATEGORICAL, features.colors.PLAYER_RELATIVE_PALETTE),
    selected=(2, features.FeatureType.CATEGORICAL, features.colors.winter),
    unit_type=(max(static_data.UNIT_TYPES) + 1, features.FeatureType.CATEGORICAL, features.colors.unit_type)
)


class CustomFeatures(features.Features):
    """
    Render feature layer from SC2 Observation protos into numpy arrays.
    Check the documentation under 'pysc2.lib.features.Features'.
    """
    def __init__(self, agent_interface_format=None, map_size=None):
        super(CustomFeatures, self).__init__(agent_interface_format, map_size)

        if self._agent_interface_format.feature_dimensions:
            pass
        if self._agent_interface_format.rgb_dimensions:
            raise NotImplementedError

    def custom_observation_spec(self):
        """Customized observation spec with spatial features."""
        obs_spec = self.observation_spec()
        aif = self._agent_interface_format
        if aif.feature_dimensions:
            obs_spec['feature_spatial'] = (
                len(SPATIAL_FEATURES),
                aif.feature_dimensions.minimap.y,
                aif.feature_dimensions.minimap.x
            )

        if aif.rgb_dimensions:
            raise NotImplementedError

        return obs_spec

    @sw.decorate
    def custom_transform_obs(self, obs):
        """Customized rendering of SC2 observations into something an agent can handle."""
        out = self.transform_obs(obs)
        aif = self._agent_interface_format

        def or_zeros(layer, size):
            if layer is not None:
                return layer.astype(np.int32, copy=False)
            else:
                return np.zeros((size.y, size.x), dtype=np.int32)

        if aif.feature_dimensions:
            out['feature_spatial'] = named_array.NamedNumpyArray(
                np.stack(or_zeros(f.unpack(obs.observation), aif.feature_dimensions.minimap) for f in SPATIAL_FEATURES),
                names=[SpatialFeatures, None, None]
            )

        if aif.rgb_dimensions:
            raise NotImplementedError

        return out


def custom_features_from_game_info(
        game_info,
        use_feature_units=False,
        use_raw_units=False,
        action_space=False,
        hide_specific_actions=True,
        use_unit_counts=False,
        use_camera_position=False):
    """
    Construct a 'CustomFeatures' object using data extracted from game info.
    Customized version of 'pysc2.lib.features.features_from_game_info'.
    """

    if game_info.options.HasField('feature_layer'):
        fl_opts = game_info.options.feature_layer
        feature_dimensions = features.Dimensions(
            screen=(fl_opts.resolution.x, fl_opts.resolution.y),
            minimap=(fl_opts.minimap_resolution.x, fl_opts.minimap_resolution.y)
        )
    else:
        feature_dimensions = None

    if game_info.options.HasField('render'):
        raise NotImplementedError
    else:
        rgb_dimensions = None

    map_size = game_info.start_raw.map_size
    camera_width_world_units = game_info.options.feature_layer.width

    return CustomFeatures(
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=feature_dimensions,
            rgb_dimensions=rgb_dimensions,
            use_feature_units=use_feature_units,
            use_raw_units=use_raw_units,
            use_unit_counts=use_unit_counts,
            use_camera_position=use_camera_position,
            camera_width_world_units=camera_width_world_units,
            action_space=action_space,
            hide_specific_actions=hide_specific_actions,
        ),
        map_size=map_size
    )
