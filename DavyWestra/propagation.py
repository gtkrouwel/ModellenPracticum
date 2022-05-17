import os
from pathlib import Path
import pandas as pd
import datetime

path_to_data = Path(os.pardir, os.pardir, "modellenpracticum2022-speed-of-heat", "data")

import sys
sys.path.append(str(path_to_data.resolve()))

def load_propagation_data(circuitnr, begin_date: datetime.date, end_date: datetime.date):
    circuitnr = str(circuitnr)

    path_to_circuitnr = Path(path_to_data, circuitnr)

    # Check if data folder exists
    if not path_to_circuitnr.is_dir():
        return "circuit number doesn't exist"

    # Create file name
    propagation_file = f"Propagation.csv"
    propagation_file_path = path_to_circuitnr / propagation_file

    # Check if the file exists
    if not propagation_file_path.is_file():
        return "propagation file doesn't exist"

    propagation_data = pd.read_csv(propagation_file_path,sep = ';')

    # Set 'Date/time (UTC)' column as datetime object and set it as index
    propagation_data["Date/time (UTC)"] = pd.to_datetime(propagation_data["Date/time (UTC)"])
    propagation_data = propagation_data.set_index("Date/time (UTC)")

    # Extract the desired time interval
    propagation_data = propagation_data["Propagation time (ns)"][begin_date:end_date]

    return propagation_data