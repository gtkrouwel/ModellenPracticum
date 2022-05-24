import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import Union

def add_to_path(dir: Path):
    """
    Adds given directory `dir` to PATH variable. Use this before importing .py
    files from a different directory, if those .py files aren't part of a
    module.

    :param dir: Path to the directory containing files to import. Note that this
    function doesn't actually import anything by itself. The path can be
    relative.
    """
    # Make path absolute and tell Python to also look there when importing.
    sys.path.append(str(dir.resolve()))

path_to_alliander_repo = Path(os.pardir, os.pardir, "modellenpracticum2022-speed-of-heat")
path_to_data = path_to_alliander_repo / "data"

# Functions for loading data.
path_to_preprocess = path_to_alliander_repo / "notebooks"
add_to_path(path_to_preprocess)
from preprocess import *

# Function to get soil temperature data.
path_to_t_soil = Path(os.pardir, "DavyWestra")
add_to_path(path_to_t_soil)
from T_soil import T_soil


current_column_heading        = 'Current'
power_column_heading          = 'Power'
reactive_power_column_heading = 'Reactive power'
cable_tempt_column_heading    = 'Cable temperature'

def get_circuit_nos() -> list[str]:
    """
    :return: List of all circuit numbers available to us.
    """
    # Have to check that it contains only digits because in the same data
    # directory there's a directory "weather_cds_data".
    # TODO remove `and subdir.name != '3249'` if the data for circuit 3249 ever
    # becomes available. (Currently the Power.csv for that circuit is basically
    # empty.)
    return [subdir.name for subdir in path_to_data.iterdir() if subdir.name.isdigit() and subdir.name != '3249']

# Dictionary.
# Key   = circuit number.
# Value = electricity data for that circuit.
_all_electricity_data = {}

def _rename_columns(column_title: str):
    """
    Helper function to pass to `pandas.DataFrame.rename`. Returns more
    human-readable version of input. Determines meaning of input by looking at
    the last two characters.

    :param column_title: Column title.
    :return: More human-readable version of `column_title`.
    """
    suffix = column_title[-2:]  # Last two characters
    if suffix == '-I':
        return current_column_heading
    if suffix == '-P':
        return power_column_heading
    if suffix == '-Q':
        return reactive_power_column_heading
    else:
        return column_title  # If not recognize, don't change.

def get_electricity_data(circuit_no: Union[int, str]) -> pd.DataFrame:
    """
    Load electricity data for a cable into a pandas dataframe. Data is cached
    because loading the data is slow. Columns (excluding the date-time column)
    are renamed to 'Current', 'Power', and 'Reactive power' (based on their
    meaning). Note that not all columns are always present.

    :param circuit_no: Circuit number of cable for which to get data.
    :return: Electricity data for cable `circuit_no`.
    """
    circuit_no = str(circuit_no)  # Make sure it's a string.
    if circuit_no not in get_circuit_nos():
        raise ValueError("Given circuit number not known.")

    # If data for `circuit_no` not yet loaded, add key-value pair with
    # key   = `circuit_no`
    # value = electricity data
    if circuit_no not in _all_electricity_data:
        # Load data and resample to 60 min.
        electricity_data = load_wop_data(circuit_no, path_to_data, True)
        # Make column titles more readable.
        electricity_data.rename(columns=_rename_columns, inplace=True)

        _all_electricity_data.update([(circuit_no, electricity_data)])
    
    return _all_electricity_data[circuit_no]

class Aux_cable_temperature_model:
    """
    Auxiliary cable temperature model. "Auxiliary" because instances of this
    class are models that compute the cable temperature based on (i) soil
    temperature and optionally (ii) electricity data. That data for the cable
    temperature is then used to optimize the parameters of the "main" model,
    which predicts cable temperature based on propagation speed.
    """

    def __init__(self, name: str, equation: str, computer):
        """
        :param name: Human-readable name describing the model.
        :param equation: Human-readable equation which the model represents.
        :param computer: Function with input (electricity data, soil temperature
        data) and output cable temperature.
        """
        self.name = name
        self.equation = equation
        self.computer = computer

    def compute_cable_temperature(self, circuit_no: Union[int, str]
    ) -> pd.core.series.Series:
        """
        :param circuit_no: ID of the cable for which cable temperature data is
        to be computed.
        :return: same as for `_compute_cable_tempt_naive()`.
        """
        circuit_no = str(circuit_no)
        electricity_data = get_electricity_data(circuit_no)

        # The time interval for which we need soil temperature data, depends on
        # the time interval for which we have electricity data.
        t_begin = electricity_data.first_valid_index()
        t_end   = electricity_data.last_valid_index()
        soil_temperature = T_soil(circuit_no, t_begin, t_end)
        current_data = electricity_data[current_column_heading]

        return self.computer(current_data, soil_temperature)


# ================================================================
# These functions do the actual computations.

def _compute_cable_tempt_naive(
    current_data: pd.core.series.Series,
    soil_temperature: pd.core.series.Series
) -> pd.core.series.Series:
    """
    Returns a series with the (predicted) cable temperature. For the output and
    all parameters, the indices are date-time objects. The intersection of the
    input is taken, meaning if a data point at time T is missing for one or both
    of the inputs, then that time T is completely ignored. As a result, the
    output may be discontinuous.
    The cable temperature is predicted according to the model:
    T_cable(t) = T_soil(t)

    :param current_data: series where values are current I.
    :param soil_temperature: series where values are soil temperature in
    Celsius.
    :return: series where values are (predicted) cable temperature.
    """
    # TODO select the data points for which correlation between propagation
    # speed and soil temperature is high, so that soil temperature equals cable
    # temperature.
    pass

def _compute_cable_tempt_linear(
    current_data: pd.core.series.Series,
    soil_temperature: pd.core.series.Series
) -> pd.core.series.Series:
    """
    Documentation exactly the same as for `_compute_cable_tempt_naive()` except
    the model used is:
    T_cable(t) = C * I(t)^2 + T_soil(t).
    """
    # The constant "C" from the model, as computed by our mathematicians.
    constant_c = 7.79e-8

    soil_tempt_column_heading = soil_temperature.name

    # Put input data in a single dataframe.
    input_data = pd.concat([current_data, soil_temperature],
        axis=1,
        join='inner'  # Intersect.
    )

    # Compute output (store in a new column next to the input data).
    input_data[cable_tempt_column_heading] = input_data.apply(lambda row :
        # T_cable(t) = C * I(t)^2 + T_soil(t).
        constant_c * row[current_column_heading]**2
        + row[soil_tempt_column_heading],
        axis=1  # Apply to each row.
    )
    return input_data[cable_tempt_column_heading]

# ================================================================
# Models.

naive_model = Aux_cable_temperature_model(
    "Naive",
    "T_cable(t) = T_soil(t)",
    _compute_cable_tempt_naive
)

linear_model = Aux_cable_temperature_model(
    "Linear",
    "T_cable(t) = C * I(t)^2 + T_soil(t)",
    _compute_cable_tempt_linear
)

# All models to try.
# TODO add naive model once it's finished (see the TODO in the body of
# `_compute_cable_tempt_naive()`).
models = [linear_model]

if __name__ == "__main__":
    # Testing.
    circuit_no = get_circuit_nos()[0]
    model = linear_model
    cable_tempt_data = model.compute_cable_temperature(circuit_no)
    print(cable_tempt_data)

#     # Example:
#     # You can do a for-loop over the models like so:
#     for circuit_no in get_circuit_nos():
#         for model in models:
#             cable_temperature_data = model.compute_cable_temperature(circuit_no)
#             # Do bayesian linear regression...
#             print("Result for circuit ", circuit_no, " using the ", model.name,
#                 " model with equation ", model.equation, ":", sep="")
#             # Show results...