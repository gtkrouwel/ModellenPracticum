import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge

import sys, os
from pathlib import Path

# Some paths needed for importing, not needed in end-product
path_to_T_soil_dir = Path(os.pardir, os.pardir, "DavyWestra")
sys.path.append(str(path_to_T_soil_dir.resolve()))
path_to_circuit_nos = Path(os.pardir, os.pardir, "Arthur")
sys.path.append(str(path_to_circuit_nos.resolve()))

from T_soil import T_soil, propagation, current
from auxiliary_cable_temperature_model import get_circuit_nos

# Input: The timeframe requested and the circuit number
# Output: Data on the soil temperature of said circuit in this time frame
def retrieve_soil_data(circuitnr, begin_date, end_date):
    return T_soil(circuitnr=circuitnr, begin_date=begin_date, end_date=end_date)

# Input: The timeframe requested and the circuit number
# Output: Data on the propagation of said circuit in this time frame
def retrieve_propagation_data(circuitnr, begin_date, end_date):
    return propagation(circuitnr=circuitnr, begin_date=begin_date, end_date=end_date)

# Input: The timeframe requested and the circuit number
# Output: Data on the current of said circuit in this time frame
def retrieve_current_data(circuitnr, begin_date, end_date):
    return current(circuitnr=circuitnr, begin_date=begin_date, end_date=end_date)

# Input: The constant c, current and soil temperature
# Output: Table temperature from the formula t_cable = c*p + t_soil
def calculate_t_cable(constant_c, curr, t_soil):
    # This works because prop and t_soil are numpy arrays
    curr_for_form = curr*curr
    t_cable = constant_c*curr_for_form + t_soil
    return t_cable

# Takes as input two datasets, the data itself and the target
# Outputs the model and the r-score of the model
def setup_bayesian_linear_regression(calc_data, calc_target):
    # Training the data to setup our model
    trained_data_calc, calc_data_test, trained_data_target, target_data_test = train_test_split(calc_data, calc_target, test_size = 0.15, random_state = 42)
    
    # Creating the model
    model = BayesianRidge()
    # Fit the trained data to the model
    model.fit(trained_data_calc, trained_data_target)

    # Model prediction on the trained data
    prediction_r_score = model.predict(calc_data_test)

    # Calculate the r-score
    r_score_model = r2_score(target_data_test, prediction_r_score)
    # print(f"r2 Score Of Test Set : {r2_score(soil_data_test, prediction_r_score)}")
    return model, r_score_model

def main():
    constant_c = 1
    begin_date, end_date = 0, 0
    circuit_nr = get_circuit_nos()
    cable_temps = np.array()
    prop_data = np.array()
    for c_nr in circuit_nr:
        curr = retrieve_current_data(c_nr, begin_date, end_date)
        t_soil = retrieve_soil_data(c_nr, begin_date, end_date)
        t_cable = calculate_t_cable(constant_c, curr, t_soil)
        prop = retrieve_propagation_data(c_nr, begin_date, end_date)
        # Fix append
        cable_temps.append([c_nr, t_cable])
        prop_data.append([c_nr, prop])
    
    for i in (0, len(cable_temps)):
        _, score = setup_bayesian_linear_regression(prop_data[i], cable_temps[i])
        print(score)
    return 0
        
if __name__ == "__main__":
    main()