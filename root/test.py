import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge
# import matplotlib.pyplot as plt
# import retrieve_model_data from .py

# This function returns the soil-temperature data, 
# Cable temperature data and the current as calculated in 2
def retrieve_data():
    # return retrieve_model_data()
    return np.zeros(shape=1), np.zeros(shape=1), np.zeros(shape=1)

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
    #temp_soil_data, temp_cable_data, current_data = retrieve_data()
    return 0
        
if __name__ == "__main__":
    main()