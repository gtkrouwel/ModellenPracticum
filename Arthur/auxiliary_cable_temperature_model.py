import os
from pathlib import Path
import numpy as np

path_to_alliander_repo = Path(os.pardir, os.pardir, "modellenpracticum2022-speed-of-heat")
path_to_data = path_to_alliander_repo / "data"
# Directory containing preprocess.py with predefined functions for loading data.
path_to_preprocess = path_to_alliander_repo / "notebooks"

import sys
# Make path absolute and tell Python to also look there when importing modules.
sys.path.append(str(path_to_preprocess.resolve()))

from preprocess import *


# List of all circuit numbers available to us.
# Have to check that it contains only digits because in the same data directory
# there's a directory "weather_cds_data".
circuit_nos = [subdir.name for subdir in path_to_data.iterdir() if subdir.name.isdigit()]

# Dictionary.
# Key   = circuit number.
# Value = electricity data for that circuit.
# 
# Note: this might take 1--2 minutes to load.
# Reason for "preloading" this data:
# We may want to use multiple models so this way prevent loading same data
# twice. (Loading data is slow).
all_electricity_data = {
    circuit_no : load_wop_data(circuit_no, path_to_data)
    for circuit_no in circuit_nos
}

# Dictionary.
# Key   = circuit number.
# Value = soil temperature data for that circuit.
all_soil_temperature_data = {}  # TODO implement. (Davy needs to put his code into .py.)

# Auxiliary cable temperature model. "Auxiliary" because instances of this class
# are models that compute the cable temperature based on (i) soil temperature
# and optionally (ii) electricity data. That data for the cable temperature is
# then used to optimize the parameters of the "main" model, which predicts cable
# temperature based on propagation speed.
class aux_cable_temperature_model:
    # `computer` is a function with input
    # - electricity data
    # - soil temperature data
    # and output:
    # - cable temperature.
    def __init__(self, name: str, equation: str, computer):
        self.name = name
        self.equation = equation
        self.computer = computer

    def compute_cable_temperature(self, circuit_no):
        electricity_data = all_electricity_data[str(circuit_no)]
        soil_temperature = all_soil_temperature_data[str(circuit_no)]
        return self.computer(electricity_data, soil_temperature)

# These functions do the actual computations.

def compute_cable_tempt_naive(electricity_data, soil_temperature):
    pass  # TODO implement

def compute_cable_tempt_linear(electricity_data, soil_temperature):
    pass  # TODO implement

# We can add models to this list.
models = [
    aux_cable_temperature_model(
          "Naive"
        , "T_cable(t) = T_soil(t)"
        , compute_cable_tempt_naive
    ),
    aux_cable_temperature_model(
          "Linear"
        , "T_cable(t) = C * P(t) + T_soil(t)"
        , compute_cable_tempt_linear
    )
]

if __name__ == "__main__":
    # Test.
    # print(electricity_data["1358"])

    # Example:
    # You can do a for-loop over the models like so:
    for circuit_no in circuit_nos:
        for model in models:
            cable_temperature_data = model.compute_cable_temperature(circuit_no)
            soil_temperature_data = all_soil_temperature_data[circuit_no]
            # Do bayesian linear regression...
            print("Result for circuit ", circuit_no, " using the ", model.name,
                " model with equation ", model.equation, ":", sep="")
            # Show results...