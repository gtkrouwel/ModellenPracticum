from datetime import datetime
from util import *
from model_confidence import get_error

# `preprocess.load_circuit_data()` suffixes this column heading with a space +
# the circuit number.
SENSITIVITY_COLUMN_HEADING = 'PD Detection Sensitivity (pC)'

def get_sensitivity_data(circuit_no: Union[int, str]) -> pd.Series:
    """
    :param circuit_no: Circuit number of the cable for which to get data.
    :return: Pandas series with datetime as index and sensitivity (pC) as
    values.
    """
    circuit_no = str(circuit_no)
    data = load_circuit_data(circuit_no, PATH_TO_DATA)
    # Set datetime as index.
    DATETIME_COLUMN_HEADING = list(data)[0]
    data = data.set_index(DATETIME_COLUMN_HEADING)

    COLUMN_HEADING = SENSITIVITY_COLUMN_HEADING + ' ' + str(circuit_no)
    sensitivity_data = data[COLUMN_HEADING]
    
    # Remove values that are zero or NaN.
    sensitivity_data = sensitivity_data.replace(0,np.nan).dropna()

    # Remove duplicates (in the .csv files some data points are written twice).
    indices_without_duplicates = ~sensitivity_data.index.duplicated()
    sensitivity_data = sensitivity_data[indices_without_duplicates]

    return sensitivity_data

def get_error_data(
        circuit_no: Union[int, str],
        absolute_error=True
    ) -> pd.Series:
    """
    :param circuit_no: Circuit number of the cable for which to get data.
    :return: Pandas series with datetime as index and error as values.
    """
    circuit_no = str(circuit_no)

    # It seems to not cause problems to use start & end times well before and
    # after the interval for which we have data.
    t_begin = datetime(2018, 1,  1)
    t_end   = datetime(2022, 6, 26)

    error_data = get_error([circuit_no], [t_begin], [t_end])[circuit_no]

    if absolute_error:
        error_data = error_data.apply(abs)

    return error_data

def get_sensitivity_and_error_data(
        circuit_no: Union[int, str],
        absolute_error=False
    ) -> pd.DataFrame:
    """
    Loads data and combines it such that only time points are included at which
    data for both "sensitivity" and "error" are available.

    :param circuit_no: Circuit number of the cable for which to get data.
    :param absolute_error: If True, take absolute value of error.
    :return: Pandas dataframe with datetime as index and as values, sensitivity
    (pC) and error.
    """
    circuit_no = str(circuit_no)
    
    combined_data = pd.concat(
        [
            get_sensitivity_data(circuit_no),
            get_error_data(circuit_no, absolute_error)
        ],
        axis=1,
        join='inner'  # Intersect.
    )
    return combined_data