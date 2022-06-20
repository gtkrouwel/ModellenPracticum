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
    # Set datatime as index.
    DATETIME_COLUMN_HEADING = list(data)[0]
    data = data.set_index(DATETIME_COLUMN_HEADING)

    COLUMN_HEADING = SENSITIVITY_COLUMN_HEADING + ' ' + str(circuit_no)
    sensitivity_data = data[COLUMN_HEADING]
    
    # Remove values that are zero or NaN.
    sensitivity_data = sensitivity_data.replace(0,np.nan).dropna()
    return sensitivity_data