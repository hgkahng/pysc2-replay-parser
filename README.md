## Overview

This repository provides the following:
- Downloading replay files (.SC2Replay) of specific StarCraft II versions
- Parsing replay files using [pysc2](https://github.com/deepmind/pysc2)

## Usage

### Step 1
- ...
### Step 2
- ...
### Step 3
- ...

## Folder Structures

    .
    +-- configs/
        +-- transform_flags.cfg
    +-- notebooks/
        +-- A-ObservedParsedFeatures.ipynb
    +-- sample/
        +-- PlayerMetaInfo.json  # Game result information
        +-- ScreenFeatures.npz   # Screen features (17)
        +-- MinimapFeatures.npz  # Minimap features (7)
        +-- SpatialFeatures.npz  # Spatial features (8 = 7 + 1)
    +-- .gitignore
    +-- custom_features.py
    +-- dowload_replays.py
    +-- parsers.py
    +-- transform_replays.py  
    +-- README.md


## Note on 'SpatialFeatures'

`SpatialFeatures` is a customized & modified version of `MinimapFeatures` implemented under [pysc2.lib.features](https://github.com/deepmind/pysc2/blob/master/pysc2/lib/features.py).

## Loading Parsed Results

Parsed results for a single replay is provided under the `sample/` directory. Check `notebooks/A-ObserveParsedFeatures.ipynb` for methods in loading them.
