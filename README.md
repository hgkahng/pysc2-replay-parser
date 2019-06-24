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
    +-- notebooks/
        +-- A-ObservedParsedFeatures.ipynb
    +-- sample/
        +-- PlayerMetaInfo.json  # Game result information
        +-- ScreenFeatures.npz   # Screen features (17)
        +-- MinimapFeatures.npz  # Minimap feature (7)
    +-- .gitignore
    +-- dowload_replays.py
    +-- replay_agent.py       
    +-- transform_replays.py  
    +-- README.md


## Loading Parsed Results

Parsed results for a single replay is provided under the 'sample' directory. Check 'notebooks/A-ObserveParsedFeatures.ipynb' for methods in loading them.
