# TODO remove commented-out code (until `from util import *`) once we're sure
# it's not necessary.

# import os
# from pathlib import Path
# import sys
# import numpy as np
# import pandas as pd
# from typing import Union

# PATH_TO_ALLIANDER_REPO = Path(os.pardir, os.pardir, 'modellenpracticum2022-speed-of-heat')
# PATH_TO_PREPROCESS = PATH_TO_ALLIANDER_REPO / 'notebooks'
# sys.path.append(str(PATH_TO_PREPROCESS.resolve()))
# # Functions for loading data.
# from preprocess import *

# PATH_TO_DATA = PATH_TO_ALLIANDER_REPO / 'data'

# CURRENT_COLUMN_HEADING        = 'Current'
# POWER_COLUMN_HEADING          = 'Power'
# REACTIVE_POWER_COLUMN_HEADING = 'Reactive power'
# CABLE_TEMP_COLUMN_HEADING     = 'Cable temperature'

# def get_circuit_nos() -> list[str]:
#     """
#     :return: List of all circuit numbers available to us.
#     """
#     # Have to check that it contains only digits because in the same data
#     # directory there's a directory "weather_cds_data".
#     # TODO remove `and subdir.name != '3249'` if the data for circuit 3249 ever
#     # becomes available. (Currently the Power.csv for that circuit is basically
#     # empty.)
#     return [subdir.name for subdir in PATH_TO_DATA.iterdir() if subdir.name.isdigit() and subdir.name != '3249']

from util import *
from temp_soil import load_temp_soil
from datetime import datetime

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
        return CURRENT_COLUMN_HEADING
    if suffix == '-P':
        return POWER_COLUMN_HEADING
    if suffix == '-Q':
        return REACTIVE_POWER_COLUMN_HEADING
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
        electricity_data = load_wop_data(circuit_no, PATH_TO_DATA, True)
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
        :return: same as for `_compute_cable_temp_naive()`.
        """
        circuit_no = str(circuit_no)
        electricity_data = get_electricity_data(circuit_no)

        # The time interval for which we need soil temperature data, depends on
        # the time interval for which we have electricity data.
        # t_begin = electricity_data.first_valid_index()  # TODO remove old code
        # t_end   = electricity_data.last_valid_index()  # TODO remove old code
        # soil_temperature = load_temp_soil(circuit_no, t_begin, t_end)  # TODO remove old code
        begin_date, end_date =  datetime(2021, 2, 21), datetime(2021, 7, 21)
        soil_temperature = load_temp_soil(circuit_no, begin_date, end_date)
        current_data = electricity_data[CURRENT_COLUMN_HEADING]

        return self.computer(current_data, soil_temperature)


# ================================================================
# These functions do the actual computations.

def _compute_cable_temp_naive(
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

def _compute_cable_temp_linear(
    current_data: pd.core.series.Series,
    soil_temperature: pd.core.series.Series
) -> pd.core.series.Series:
    """
    Documentation exactly the same as for `_compute_cable_temp_naive()` except
    the model used is:
    T_cable(t) = C * I(t)^2 + T_soil(t).
    """
    # The constant "C" from the model, as computed by our mathematicians.
    constant_c = 7.79e-8

    soil_temp_column_heading = soil_temperature.name

    # Put input data in a single dataframe.
    input_data = pd.concat([current_data, soil_temperature],
        axis=1,
        join='inner'  # Intersect.
    )

    # Compute output (store in a new column next to the input data).
    input_data[CABLE_TEMP_COLUMN_HEADING] = input_data.apply(lambda row :
        # T_cable(t) = C * I(t)^2 + T_soil(t).
        constant_c * row[CURRENT_COLUMN_HEADING]**2
        + row[soil_temp_column_heading],
        axis=1  # Apply to each row.
    )
    return input_data[CABLE_TEMP_COLUMN_HEADING]

# ================================================================
# Models.

naive_model = Aux_cable_temperature_model(
    "Naive",
    "T_cable(t) = T_soil(t)",
    _compute_cable_temp_naive
)

linear_model = Aux_cable_temperature_model(
    "Linear",
    "T_cable(t) = C * I(t)^2 + T_soil(t)",
    _compute_cable_temp_linear
)

# All models to try.
# TODO add naive model once it's finished (see the TODO in the body of
# `_compute_cable_temp_naive()`).
MODELS = [linear_model]

if __name__ == "__main__":
    # Testing.
    circuit_no = get_circuit_nos()[0]
    model = linear_model
    cable_temp_data = model.compute_cable_temperature(circuit_no)
    print(cable_temp_data)

#     # Example:
#     # You can do a for-loop over the MODELS like so:
#     for circuit_no in get_circuit_nos():
#         for model in MODELS:
#             cable_temperature_data = model.compute_cable_temperature(circuit_no)
#             # Do bayesian linear regression...
#             print("Result for circuit ", circuit_no, " using the ", model.name,
#                 " model with equation ", model.equation, ":", sep="")
#             # Show results...