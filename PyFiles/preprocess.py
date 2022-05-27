import pandas as pd
from pathlib import Path


def load_circuit_data(circuitnr, path: str):
    """
    Load the csv data (propagation, sensitively & pd) from the circuit directorys (and resamples PD to 60min)
    :param circuitnr: List of circuits to load
    :param path: Data directory path
    :return dataframe with added column(s)
    """
    circ_dir = Path(path) / str(circuitnr)

    df1 = pd.read_csv(circ_dir / 'Propagation.csv', sep=';', parse_dates=['Date/time (UTC)'])
    df1 = df1.rename(columns={"Propagation time (ns)": "Propagation time (ns) " + str(circuitnr)})

    df2 = pd.read_csv(circ_dir / 'Sensitivity.csv', sep=';', parse_dates=['Date/time (UTC)'])
    df2 = df2.rename(columns={"PD Detection Sensitivity (pC)": "PD Detection Sensitivity (pC) " + str(circuitnr)})

    df3 = pd.read_csv(circ_dir / 'PD.csv', sep=';', parse_dates=['Date/time (UTC)'])
    df3 = df3.drop("Location in meters (m)", axis=1)
    df3 = df3.rename(columns={"Charge (picocoulomb)": "Total charge (pC) " + str(circuitnr)})
    df3_res = df3.resample('60min', on='Date/time (UTC)').sum()

    df = pd.merge(df1, df2, on="Date/time (UTC)", how='left')
    df = pd.merge(df, df3_res, on="Date/time (UTC)", how='left')
    return df


def load_wop_data(circuitnr, path: str, resample=False):
    """
    Load the csv data (power, current & voltage) from WOP datadump and resample to 60min
    :param path: Data directory path
    :param resample: resample data to 60min if true
    :return dataframe with added column(s)
    """
    circ_dir = Path(path) / str(circuitnr)
    power_df = pd.read_csv(circ_dir / 'Power.csv', sep=';', parse_dates={'Date/time (UTC)': [' Datum', 'Tijd']},
                           date_parser=(lambda x: pd.to_datetime(x, format="%Y/%m/%d %H:%M")), decimal=',')

    power_df.drop(power_df.columns[power_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if resample:
        power_df = power_df.resample('60min', on='Date/time (UTC)').mean()
    return power_df

def load_combined_data(circuitnr, path: str):
    wop_data = load_wop_data(circuitnr, path, True)
    circuit_data = load_circuit_data(circuitnr, path).resample('60min', on='Date/time (UTC)').mean()
    return pd.merge(wop_data, circuit_data, on="Date/time (UTC)", how='left')
