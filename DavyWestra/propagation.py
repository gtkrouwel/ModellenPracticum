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



def load_circuit_data(circuitnr, path: str):
    """
    Load the csv data (propagation, sensitively & pd) from the circuit directorys (and resamples PD to 60min)
    :param circuitnr: List of circuits to load
    :param path: Data directory path
    :return dataframe with added column(s)
    """
    circ_dir = Path(path) / str(circuitnr)

    # Check if data folder exists
    if not path_to_circuitnr.is_dir():
        return "circuit number doesn't exist"

    # Load propagation data
    df1 = pd.read_csv(circ_dir / 'Propagation.csv', sep=';', parse_dates=['Date/time (UTC)'])
    df1 = df1.rename(columns={"Propagation time (ns)": "Propagation time (ns) " + str(circuitnr)})

    # Load sensitivity data
    df2 = pd.read_csv(circ_dir / 'Sensitivity.csv', sep=';', parse_dates=['Date/time (UTC)'])
    df2 = df2.rename(columns={"PD Detection Sensitivity (pC)": "PD Detection Sensitivity (pC) " + str(circuitnr)})

    # Load PD data
    df3 = pd.read_csv(circ_dir / 'PD.csv', sep=';', parse_dates=['Date/time (UTC)'])
    df3 = df3.drop("Location in meters (m)", axis=1)
    df3 = df3.rename(columns={"Charge (picocoulomb)": "Total charge (pC) " + str(circuitnr)})
    df3_res = df3.resample('60min', on='Date/time (UTC)').sum()

    df = pd.merge(df1, df2, on="Date/time (UTC)", how='left')
    df = pd.merge(df, df3_res, on="Date/time (UTC)", how='left')

    # Set 'Date/time (UTC)' column as datetime object and set it as index
    df["Date/time (UTC)"] = pd.to_datetime(df["Date/time (UTC)"])
    df = df.set_index("Date/time (UTC)")

    return df