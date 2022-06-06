from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from pandas import concat
from numpy import isnan, delete
from datetime import datetime

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
# Output: Table temperature from the formula t_cable = c*p + t_soil
def calculate_t_cable(curr, t_soil):
    # This works because prop and t_soil are numpy arrays
    return CONSTANT_C*(curr**2) + t_soil

# Takes as input two datasets, the data itself and the target
# Outputs the model and the r-score of the model
def setup_bayesian_linear_regression(calc_data, calc_target):
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
    model = LinearRegression()
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

    print("Shape of unfiltered data: " + str(cat_data.shape))

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
    print("Shape of filtered data: " + str(filtered_data.shape))
    print("Correlation is: " +  str(corr) + " and threshold is: " + str(threshold))
    return filtered_data

# Takes as input the whole data-set for a circuit and the soil, but also whether we filter the load
# Outputs all data combined by correct date
def shape_data(comb_data, temp_soil, low_load):
    # Concatenate the data
    cat_data = concat([comb_data, temp_soil], axis=1, join='inner')

    # Filter the data if the flag is set
    if low_load:
        cat_data = filter_data(cat_data)
    cat_data = cat_data.to_numpy()

    # Some circuit miss some columns, this solves the issue
    _, no_of_columns = cat_data.shape
    type_indexing = 7 - no_of_columns
    return cat_data[:,0], cat_data[:,3-type_indexing], cat_data[:,6-type_indexing]

# Input is the a1 as determined by the low load cases and the data for a circuit (so prop, t_soil and curr)
# Output is the constant C
def reverse_engineer_c(a1, prop, t_soil, curr):
    # Calculate t_cable through t_cable = a0 + a1*prop
    t_cable = a1*prop

    # Change the model to T_cable - T_soil = C*I^2(t)
    target_data = t_cable - t_soil

    # Linear regression on this model, which determines C
    model, _ = setup_bayesian_linear_regression(calc_data=(curr**2), calc_target=target_data)
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
        t_soil = retrieve_soil_data(c_nr, begin_date, end_date)
        
        # Fix the shapes of the data
        curr, prop, t_soil = shape_data(comb_data, t_soil, low_load)

        # Calculate the cable temperature
        if low_load:
            t_cable = t_soil
        else:
            t_cable = calculate_t_cable(curr, t_soil)

        # Actually does the linear regression
        model, score = setup_bayesian_linear_regression(prop, t_cable)
        model_list.append([c_nr, model, score])

        # Premature printing
        #print(model.coef_) 
        #print(score)

        # If the flag is set, we want to calculate C 
        if low_load:
            constant_c = reverse_engineer_c(model.coef_[0], prop, t_soil, curr)
            print ("Constant_c is: " + str(constant_c))
    return model_list

def main():
    # Start and end dates
    begin_date, end_date =  datetime(2021, 2, 21), datetime(2021, 7, 21)
    
    # Retrieving all valid circuit numbers
    circuit_nr = get_circuit_nos()
    circuit_nr.remove("2821")

    # Variable to filter low-load or not
    low_load = True

    # Function which calculates the alpha's
    model_list = confidence(circuit_nr, begin_date, end_date, low_load)
    print("Now the summary: ")
    for model in model_list:
        print("The model for circuit number " + str(model[0]) + ":")
        print("The coefficients for a0 and a1 are 0 and " + str(model[1].coef_[0]))
        print("The confidence is: " + str(model[2]))
    return 0
        
if __name__ == "__main__":
    main()