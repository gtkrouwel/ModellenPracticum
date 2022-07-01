import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import Union

PATH_TO_ALLIANDER_REPO = Path(
    os.pardir,
    os.pardir,
    'modellenpracticum2022-speed-of-heat'
)
PATH_TO_PREPROCESS = PATH_TO_ALLIANDER_REPO / 'notebooks'
sys.path.append(str(PATH_TO_PREPROCESS.resolve()))
# Functions for loading data.
from preprocess import *

PATH_TO_DATA = PATH_TO_ALLIANDER_REPO / 'data'

# CURRENT_COLUMN_HEADING        = 'Current'
# POWER_COLUMN_HEADING          = 'Power'
# REACTIVE_POWER_COLUMN_HEADING = 'Reactive power'
# CABLE_TEMP_COLUMN_HEADING     = 'Cable temperature'

def get_circuit_nos() -> list[str]:
    """
    :return: List of all circuit numbers available to us.
    """
    # Have to check that it contains only digits because in the same data
    # directory there's a directory "weather_cds_data".
    # TODO remove `and subdir.name != '3249'` if the data for circuit 3249 ever
    # becomes available. (Currently the Power.csv for that circuit is basically
    # empty.)
    return [subdir.name for subdir in PATH_TO_DATA.iterdir() if subdir.name.isdigit() and subdir.name != '3249']