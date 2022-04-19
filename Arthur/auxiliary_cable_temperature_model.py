import os
from pathlib import Path
import numpy as np
from copy import deepcopy

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
_circuit_nos = [subdir.name for subdir in path_to_data.iterdir() if subdir.name.isdigit()]

# Returns list of circuit numbers. This list can be edited and afterwards this
# function will still return the correct list.
def get_circuit_nos():
    return deepcopy(_circuit_nos)

# Dictionary.
# Key   = circuit number.
# Value = electricity data for that circuit.
_all_electricity_data = {}
# TODO remove old code later:
# _all_electricity_data = {
#     circuit_no : load_wop_data(circuit_no, path_to_data)
#     for circuit_no in _circuit_nos
# }

# Returns multidimensional array (TODO find out what kind, numpy? pandas dataframe?)
# of electricity data. Data is cached because loading the data is slow.
def get_electricity_data(circuit_no):
    circuit_no = str(circuit_no)  # Make sure it's a string and not a number.

    # If data for `circuit_no` not yet loaded, add key-value pair with
    # key   = circuit_no
    # value = electricity data
    if circuit_no not in _all_electricity_data:
        _all_electricity_data.update([(circuit_no, load_wop_data(circuit_no, path_to_data))])
    
    return _all_electricity_data[circuit_no]

# Dictionary.
# Key   = circuit number.
# Value = soil temperature data for that circuit.
all_soil_temperature_data = {}  # TODO implement.

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
        circuit_no = str(circuit_no)
        electricity_data = get_electricity_data(circuit_no)
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
    for circuit_no in _circuit_nos:
        for model in models:
            cable_temperature_data = model.compute_cable_temperature(circuit_no)
            soil_temperature_data = all_soil_temperature_data[circuit_no]
            # Do bayesian linear regression...
            print("Result for circuit ", circuit_no, " using the ", model.name,
                " model with equation ", model.equation, ":", sep="")
            # Show results...