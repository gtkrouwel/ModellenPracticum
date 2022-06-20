from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import bayes_mvs
from pandas import concat
from numpy import isnan, delete
from datetime import datetime
from tabulate import tabulate

from temp_soil import load_temp_soil
from preprocess import load_combined_data
from auxiliary_cable_temperature_model import get_circuit_nos
import os
from pathlib import Path

# Constant C as determined outside of this program
CONSTANT_C = 7.79e-8
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
# Output: Table temperature from the formula temp_cable = c*p + temp_soil
def calculate_temp_cable(constant_c, curr, temp_soil):
    # This works because prop and temp_soil are numpy arrays
    return constant_c*(curr**2) + temp_soil

# Input: A list of circuit numbers, start and end date and optional low load (but should always be true)
# Output: An associative array where the circuit number is the index and the values are the errors
def get_error(c_nr, begin_date, end_date, low_load=True):
    #TODO: Potentially add all times, ie create pd frame
    #TODO: Remove NANs from the data
    model_list = confidence(c_nr, begin_date, end_date, low_load)
    error = {}
    for model in model_list:
        error.update({model[0]:model[4]})
    return error

# Takes as input two datasets, the data itself and the target
# Outputs the model and the r-score of the model
def setup_bayesian_linear_regression(calc_data, calc_target, fit_intercept=True):
    # This part removes all NANs from the data
    list_index = []
    for i in range(0, len(calc_data)):
        if isnan(calc_data[i]) or isnan(calc_target[i]):
            list_index.append(i)
    calc_data = delete(calc_data, list_index)
    calc_target = delete(calc_target, list_index)

    # Training the data to setup our model
    trained_data_calc, test_data_calc, trained_data_target, test_data_target = train_test_split(calc_data, calc_target)
    # Creating the model
    model = LinearRegression(fit_intercept=fit_intercept)
    # Fit the trained data to the model
    model.fit(trained_data_calc.reshape(-1, 1), trained_data_target)

    # Model prediction on the trained data
    prediction_r_score = model.predict(test_data_calc.reshape(-1, 1))

    # Calculate the r-score
    r_score_model = r2_score(test_data_target, prediction_r_score)
    # print(f"r2 Score Of Test Set : {r2_score(soil_data_test, prediction_r_score)}")
    return model, r_score_model

# Input is all data (so prop, curr and temp_soil)
# Output is all data where the load is low
def filter_data(cat_data):
    threshold, corr = 0.0, 0.0
    # Fixing the missing columns
    _, no_of_columns = cat_data.shape
    type_indexing = 7 - no_of_columns

    # Make the currents iterable
    loads = set(cat_data.iloc[:,0])

    # Loop through all currents
    for load in loads:
        # Filter data less than the load
        filtered_data = cat_data[cat_data.iloc[:,0] <= load]

        # Calculate the correlation coefficient
        new_corr = filtered_data.iloc[:,6-type_indexing].corr(filtered_data.iloc[:,3-type_indexing], method="pearson", min_periods=500)
        
        # If this correlation is higher, than the threshold should change
        if new_corr > corr:
            corr = new_corr
            threshold = load
    
    # Only take the data below the threshold for the current 
    filtered_data = cat_data[cat_data.iloc[:,0] <= threshold]
    return filtered_data

# Takes as input the whole data-set for a circuit and the soil, but also whether we filter the load
# Outputs all data combined by correct date
def shape_data(comb_data, temp_soil):
    # Concatenate the data
    cat_data = concat([comb_data, temp_soil], axis=1, join='inner')
    cat_data_low_load = cat_data

    # Filter the data if the flag is set
    cat_data_low_load = filter_data(cat_data_low_load).to_numpy()
    cat_data = cat_data.to_numpy()

    # Some circuit miss some columns, this solves the issue
    type_indexing = 7 - cat_data.shape[1]
    cat_data = (cat_data[:,0], cat_data[:,3-type_indexing], cat_data[:,6-type_indexing])
    cat_data_low_load = (cat_data_low_load[:,0], cat_data_low_load[:,3-type_indexing], cat_data_low_load[:,6-type_indexing])

    return cat_data, cat_data_low_load

# Input is the a1 as determined by the low load cases and the data for a circuit (so prop, temp_soil and curr)
# Output is the constant C
def reverse_engineer_c(a0, a1, prop, temp_soil, curr):
    # Calculate temp_cable through temp_cable = a0 + a1*prop
    temp_cable = a0 + a1*prop

    # Change the model to temp_cable - temp_soil = C*I^2(t)
    target_data = temp_cable - temp_soil

    # Linear regression on this model, which determines C
    model, _ = setup_bayesian_linear_regression(calc_data=(curr**2), calc_target=target_data, fit_intercept=False)
    return model.coef_[0]

# Input is the circuit number list, time frame and a flag for low_load
# Output is a model for each circuit in an array
def confidence(circuit_nr, begin_date, end_date, low_load):
    model_list = []
    for c_nr in circuit_nr:
        print("Investigating circuit: " + str(c_nr))
        # Retrieving current and prop data in a single frame
        comb_data = retrieve_combined_data(c_nr, begin_date, end_date)
        # Retrieving soil temp in a single frame
        temp_soil = retrieve_soil_data(c_nr, begin_date, end_date)
        
        # Fix the shapes of the data
        cat_data, cat_data_low_load = shape_data(comb_data, temp_soil)

        # Calculate the cable temperature
        if low_load:
            curr, prop, temp_soil = cat_data_low_load
            prop = 1/prop
            temp_cable = temp_soil
        # This else-statement is not really needed
        else:
            curr, prop, temp_soil = cat_data
            prop = 1/prop
            temp_cable = calculate_temp_cable(CONSTANT_C, curr, temp_soil)

        # Actually does the linear regression
        model, score = setup_bayesian_linear_regression(prop, temp_cable)
        model_list.append([c_nr, model, score])

        # If the flag is set, we want to calculate C 
        if low_load:
            # Change the data to all points
            curr, prop, temp_soil = cat_data
            prop = 1/prop

            # Calculate (a0 + a1*prop) - T_soil = C * I^2
            constant_c = reverse_engineer_c(model.intercept_, model.coef_[0], prop, temp_soil, curr)
            print ("Constant_c is: " + str(constant_c))

            # Recalculate a1 with new constant C
            temp_cable = calculate_temp_cable(constant_c, curr, temp_soil)
            model, score = setup_bayesian_linear_regression(prop, temp_cable)

            # Calculates the error for question 4
            epsilon_error = temp_cable - (model.intercept_ + model.coef_[0]*prop)
            model_list.pop()
            model_list.append([c_nr, model, score, constant_c, epsilon_error])
    return model_list

def main():
    # Start and end dates
    begin_date, end_date =  datetime(2021, 2, 21), datetime(2021, 7, 21)
    
    # Retrieving all valid circuit numbers
    circuit_nr = get_circuit_nos()
    circuit_nr.remove("2821")

    # Variable to filter low-load or not
    low_load = True

    col_names = ["Circuit number", "a0", "a1", "confidence", "constant_c"]

    # Function which calculates the alpha's
    model_list = confidence(circuit_nr, begin_date, end_date, low_load)
    print("Now the summary: ")
    for model in model_list:
        print("The model for circuit number " + str(model[0]) + ":")
        print("The coefficient a0 is: " + str(model[1].intercept_) )
        print("The coefficient a1 is: " + str(model[1].coef_[0]))
        print("The confidence is: " + str(model[2]))
        if low_load:
            print("The constant C is: " + str(model[3]))
            print("The error is: " + str(model[4]) )
    table = [[model[0], model[1].intercept_, model[1].coef_[0], model[2], model[3]] for model in model_list]
    print(tabulate(table, col_names, tablefmt="fancy_grid"))
    return 0
        
if __name__ == "__main__":
    main()