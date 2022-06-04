from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge, LinearRegression
import pandas as pd
from numpy import isnan, delete
from datetime import datetime

from temp_soil import load_temp_soil
from preprocess import load_combined_data
from auxiliary_cable_temperature_model import get_circuit_nos
import os
from pathlib import Path

PATH_TO_DATA = Path(os.pardir, os.pardir, "modellenpracticum2022-speed-of-heat") / "data"

# Input: The timeframe requested and the circuit number
# Output: Data on the soil temperature of said circuit in this time frame
def retrieve_soil_data(circuitnr, begin_date, end_date):
    return load_temp_soil(circuitnr, begin_date, end_date)

# Input: The timeframe requested and the circuit number
# Output: Data on the current and propagation of said circuit in this time frame
def retrieve_combined_data(circuitnr, begin_date, end_date):
    return load_combined_data(circuitnr, PATH_TO_DATA)[begin_date:end_date]

# Input: The constant c, current and soil temperature
# Output: Table temperature from the formula t_cable = c*p + t_soil
def calculate_t_cable(constant_c, curr, t_soil):
    # This works because prop and t_soil are numpy arrays
    return constant_c*(curr**2) + t_soil

# Takes as input two datasets, the data itself and the target
# Outputs the model and the r-score of the model
def setup_bayesian_linear_regression(calc_data, calc_target):
    list_index = []
    for i in range(0, len(calc_data)):
        if isnan(calc_data[i]) or isnan(calc_target[i]):
            list_index.append(i)

    calc_data = delete(calc_data, list_index)
    calc_target = delete(calc_target, list_index)

    # Training the data to setup our model
    trained_data_calc, test_data_calc, trained_data_target, test_data_target = train_test_split(calc_data, calc_target)
    # Creating the model
    model = LinearRegression()
    # Fit the trained data to the model
    model.fit(trained_data_calc.reshape(-1, 1), trained_data_target)

    # Model prediction on the trained data
    prediction_r_score = model.predict(test_data_calc.reshape(-1, 1))

    # Calculate the r-score
    r_score_model = r2_score(test_data_target, prediction_r_score)
    # print(f"r2 Score Of Test Set : {r2_score(soil_data_test, prediction_r_score)}")
    return model, r_score_model

def main():
    constant_c = 7.79e-8
    begin_date, end_date =  datetime(2021, 2, 21), datetime(2022, 2, 21)
    circuit_nr = get_circuit_nos()
    circuit_nr.remove("2821")
    # For each circuit number, retrieve all required data
    for c_nr in circuit_nr:
        comb_data = retrieve_combined_data(c_nr, begin_date, end_date)
        t_soil = retrieve_soil_data(c_nr, begin_date, end_date)
        input_data = pd.concat([comb_data, t_soil], axis=1, join='inner').to_numpy()
        
        _, no_of_columns = input_data.shape
        type_indexing = 7 - no_of_columns
        curr = input_data[:,0]
        prop = input_data[:,3-type_indexing]
        t_soil = input_data[:,6-type_indexing]
        t_cable = calculate_t_cable(constant_c, curr, t_soil)

        model, score = setup_bayesian_linear_regression(prop, t_cable)
        print(model.coef_)
        print(score)
    return 0
        
if __name__ == "__main__":
    main()