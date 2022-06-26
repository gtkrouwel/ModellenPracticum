from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import bayes_mvs
from pandas import concat, Series
from numpy import isnan, delete
from datetime import datetime
from tabulate import tabulate

from temp_soil import load_temp_soil
from preprocess import load_combined_data
from util import get_circuit_nos
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

# Sadly, this helper function has to be hard-coded for now
# Output: All dates corresponding to a circuit
def get_dates():
    all_dates = {}
    # Circuit 1358
    begin_date, end_date =  datetime(2021, 1, 1), datetime(2022, 2, 1)
    all_dates.update({"1358": (begin_date, end_date)})

    # Circuit 2003
    begin_date, end_date =  datetime(2021, 1, 1), datetime(2021, 7, 1)
    all_dates.update({"2003": (begin_date, end_date)})

    # Circuit 20049
    begin_date, end_date =  datetime(2020, 7, 16), datetime(2022, 3, 1)
    all_dates.update({"20049": (begin_date, end_date)})

    # Circuit 20726
    begin_date, end_date =  datetime(2020, 3, 6), datetime(2022, 3, 1)
    all_dates.update({"20726": (begin_date, end_date)})

    # Circuit 22102
    begin_date, end_date =  datetime(2021, 2, 23), datetime(2022, 3, 1)
    all_dates.update({"22102": (begin_date, end_date)})

    # Circuit 2308
    begin_date, end_date =  datetime(2020, 12, 3), datetime(2022, 3, 1)
    all_dates.update({"2308": (begin_date, end_date)})

    # Circuit 2611
    begin_date, end_date =  datetime(2019, 10, 16), datetime(2022, 3, 1)
    all_dates.update({"2611": (begin_date, end_date)})

    # Circuit 3410
    begin_date, end_date =  datetime(2021, 1, 1), datetime(2022, 3, 1)
    all_dates.update({"3410": (begin_date, end_date)})

    # Circuit 3512
    begin_date, end_date =  datetime(2019, 10, 8), datetime(2022, 3, 1)
    all_dates.update({"3512": (begin_date, end_date)})

    # Circuit 3543
    begin_date, end_date =  datetime(2021, 4, 13), datetime(2022, 3, 1)
    all_dates.update({"3543": (begin_date, end_date)})

    return all_dates

# Input: A list of circuit numbers, list of start dates & list of end dates and optional low load (but should always be true)
# Output: An associative array where the circuit number is the index and the values are pandas.Series of error data indexed by datetime
def get_error(c_nr, begin_date, end_date, low_load=True):
    # Change data-type input of the dates
    all_dates = {}
    for i in range(0, len(begin_date)):
        all_dates.update({c_nr[i]: (begin_date[i], end_date[i])})

    # Calculate the model
    model_list = confidence(c_nr, all_dates, low_load)

    # Output the dates and errors
    error = {}
    for model in model_list:
        datetime_indices   = model[3]
        error_data         = model[5]
        indexed_error_data = Series(data=error_data, index=datetime_indices, name='error')
        error.update({model[0]:indexed_error_data})
    return error

# Takes as input two datasets, the data itself and the target
# Outputs the model and the r-score of the model
def setup_bayesian_linear_regression(calc_data, calc_target, fit_intercept=True):
    # This part removes all NANs from the data
    list_index = []
    for i in range(0, len(calc_data)):
        # If either of the sets is NAN, remove it from both sets
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

# Takes as input the whole data-set for a circuit and the soil
# Outputs low load data, unfiltered data and the dates of the unfiltered data
def shape_data(comb_data, temp_soil):
    # Concatenate the data
    cat_data = concat([comb_data, temp_soil], axis=1, join='inner')
    cat_data_low_load = cat_data

    # The dates left are exactly the indices of cat_data
    filtered_dates = cat_data.index

    # Filter the data based on low load
    cat_data_low_load = filter_data(cat_data_low_load).to_numpy()
    cat_data = cat_data.to_numpy()

    # Some circuit miss some columns, this solves the issue
    type_indexing = 7 - cat_data.shape[1]
    cat_data = (cat_data[:,0], cat_data[:,3-type_indexing], cat_data[:,6-type_indexing])
    cat_data_low_load = (cat_data_low_load[:,0], cat_data_low_load[:,3-type_indexing], cat_data_low_load[:,6-type_indexing])

    return cat_data, cat_data_low_load, filtered_dates

# Input is the a0 and a1 as determined by the low load cases and the data for a circuit (so prop, temp_soil and curr)
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
# Output is a model for each circuit in the circuit number array
def confidence(circuit_nr, all_dates, low_load):
    model_list = []

    for c_nr in circuit_nr:
        # The dates are stored per circuit
        begin_date, end_date = all_dates[c_nr]

        # Retrieving current and prop data in a single frame
        comb_data = retrieve_combined_data(c_nr, begin_date, end_date)
        # Retrieving soil temp in a single frame
        temp_soil = retrieve_soil_data(c_nr, begin_date, end_date)
        
        # Fix the shapes of the data
        cat_data, cat_data_low_load, filtered_dates = shape_data(comb_data, temp_soil)

        # Calculate the cable temperature
        if low_load:
            curr, prop, temp_soil = cat_data_low_load
            prop = 1/prop
            temp_cable = temp_soil
        # This else-statement is needed if and only if the constant C can be correctly calculated outside of this program
        else:
            curr, prop, temp_soil = cat_data
            prop = 1/prop
            temp_cable = calculate_temp_cable(CONSTANT_C, curr, temp_soil)

        # Actually does the linear regression
        model, score = setup_bayesian_linear_regression(prop, temp_cable)
        model_list.append([c_nr, model, score, filtered_dates])

        # If the flag is set, we want to calculate C 
        if low_load:
            # Change the data to all points
            curr, prop, temp_soil = cat_data
            prop = 1/prop

            # Calculate (a0 + a1*prop) - T_soil = C * I^2, from which we retrieve the constant C
            constant_c = reverse_engineer_c(model.intercept_, model.coef_[0], prop, temp_soil, curr)

            # Recalculate a1 with new constant C
            # The model is: temp_cable = C*I^2(t) + temp_soil
            temp_cable = calculate_temp_cable(constant_c, curr, temp_soil)
            # Now we fit: temp_cable = a0 + a1*prop + e(t)
            model, score = setup_bayesian_linear_regression(prop, temp_cable)

            # Calculates the error for question 4, as this is the difference between the model with C and the model with a0 and a1
            epsilon_error = temp_cable - (model.intercept_ + model.coef_[0]*prop)

            # Calculate the confidence interval on the mean, variance and std based on bayesian linear regression
            mean, var, std = bayes_mvs(temp_cable, alpha=score)

            # Clean up the list
            model_list.pop()
            model_list.append([c_nr, model, score, filtered_dates, constant_c, epsilon_error, (mean, var, std)])

    return model_list

def main():
    # Start and end dates
    all_dates = get_dates()
    
    # Retrieving all valid circuit numbers
    circuit_nr = get_circuit_nos()

    # Circuit number 2821 seems to be missing data, hence we remove it
    circuit_nr.remove("2821")

    # Boolean to filter low-load or not
    low_load = True

    # Function which calculates the alpha's
    model_list = confidence(circuit_nr, all_dates, low_load)

    # Now printing the final results
    print("Now the summary: ")
    for model in model_list:
        print("The model for circuit number " + str(model[0]) + ":")
        print("The coefficient a0 is: " + str(model[1].intercept_) )
        print("The coefficient a1 is: " + str(model[1].coef_[0]))
        print("The confidence is: " + str(model[2]))
        # If the low-load flag was set, we have way more ouput
        if low_load:
            print("The constant C is: " + str(model[4]))
            print("The error is: " + str(model[5]))
            print("The confidence interval for temp_cable is (a0 + a1*prop - error, a0 + a1*prop + error) = (" + str(model[1].intercept_ - max(model[5])) + " + " + str(model[1].coef_[0]) + " * prop - " + ", " + str(model[1].intercept_ + max(model[5])) + " + " + str(model[1].coef_[0]) + " * prop + " + ")")
            print("The average interval for cable temp is " + str(model[6][0]) + " and its variance is " + str(model[6][1]))

    # Convenient table printing for the report, not necessary
    # First table consists of circuit number, a0 and a1, the r-score and the constant c
    col_names = ["Circuit number", "a0", "a1", "confidence", "constant_c"]
    table = [[model[0], model[1].intercept_, model[1].coef_[0], model[2], model[4]]  for model in model_list]

    # Second table consists of circuit number and the confidence interval around mean and variance
    col_names_2 = ["Circuit number", "mean_confidence", "var_confidence"]
    table_2 = [[model[0], model[6][0][1], model[6][1][1]] for model in model_list]

    # Printing the tables
    print(tabulate(table, col_names, tablefmt="fancy_grid"))
    print(tabulate(table_2, col_names_2, tablefmt="fancy_grid"))

    return 0
        
if __name__ == "__main__":
    main()