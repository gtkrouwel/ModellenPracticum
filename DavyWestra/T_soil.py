#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyproj import Transformer

import datetime
import logging
from pathlib import Path

import pandas as pd
import requests


#circuits coordinates
circuit_1358 = 131436.240,502678.470
circuit_2003 = 117955.218,479433.147
circuit_2308 = 105884.568,465322.411
circuit_2611 = 111530.098,516758.349
circuit_2821 = 114053.548,471692.944
circuit_3249 = 163329.228,564016.874
circuit_3410 = 132660.045,501028.933
circuit_3512 = 198110.621,585155.202
circuit_3543 = 106754.478,500162.533
circuit_20049 = 165236.264,577272.523
circuit_20726 = 206863.49,567024.869
circuit_22102 = 158361.29,432857.689



DATA_PATH = "data"

logger = logging.getLogger(__name__)

data_path = Path(DATA_PATH)
downloaded_weather_folder = data_path / "weather_data"
downloaded_weather_uur_folder = data_path / "weather_uur_data"
downloaded_weather_cds_folder = data_path / "weather_cds_data"

# input X and Y as a tuple
def XY_to_latlon(XY):
    # transforms from Dutch Coordinate System (Amersfoort / RD New) to World Coordinate System (WGS84)
    transformer = Transformer.from_crs( "EPSG:28992","EPSG:4326")
    lat, lon = transformer.transform(XY[0],XY[1])
    return lat,lon

def _format_date(unformatted_date: datetime.date):
    return unformatted_date.strftime("%Y-%m-%d")


def _create_url(begin_date: datetime.date, end_date: datetime.date, lat=52.1, lon=5.18):
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)
    url_template = f"https://weather.appx.cloud/api/v2/weather/sources/knmi/models/daggegevens?begin={begin_date_str}&end={end_date_str}&lat={lat}&lon={lon}&units=human&response_format=csv"
    return url_template


def _create_url_uur(begin_date: datetime.date, end_date: datetime.date, lat=52.1, lon=5.18):
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)
    url_template = f"https://weather.appx.cloud/api/v2/weather/sources/knmi/models/uurgegevens?begin={begin_date_str}&end={end_date_str}&lat={end_date_str}&lat={str(lat)}&lon={str(lon)}&units=human&response_format=csv"
    return url_template


def _create_url_cds(begin_date: datetime.date, end_date: datetime.date, lat=52.1, lon=5.18):
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)
    url_template = f"https://weather.appx.cloud/api/v2/weather/sources/cds/models/era5sl?begin={begin_date_str}&end={end_date_str}&lat={lat}&lon={lon}&units=human&response_format=csv"
    return url_template


def _download_csv_from_url(url: str, weather_file_path: Path):
    download_res = requests.get(url)
    download_res.raise_for_status()  # Crash het programma als de download mislukte

    with open(weather_file_path, "wb") as f:
        f.write(download_res.content)


def load_weather_data(begin_date: datetime.date, end_date: datetime.date):
    # Check if weather folder exists
    if not downloaded_weather_folder.is_dir():
        downloaded_weather_folder.mkdir()

    # Format dates
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)

    # Create file name
    weather_file = f"knmi_daggegevens_{begin_date_str}_{end_date_str}.csv"
    weather_file_path = downloaded_weather_folder / weather_file  # pathlib.Path object

    # Check if the file exists
    if not weather_file_path.is_file():
        logger.info("Weather data not found in cache. Downloading new data.")
        download_url = _create_url(begin_date, end_date)
        _download_csv_from_url(download_url, weather_file_path)

    weather_df = pd.read_csv(weather_file_path)
    return weather_df


def load_weather_data_uur(begin_date: datetime.date, end_date: datetime.date, lat=52.1, lon=5.18):
    # Check if weather folder exists
    if not downloaded_weather_uur_folder.is_dir():
        downloaded_weather_uur_folder.mkdir()

    # Format dates
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)

    # Create file name
    weather_file = f"knmi_uurgegevens_{begin_date_str}_{end_date_str}.csv"
    weather_file_path = downloaded_weather_uur_folder / weather_file  # pathlib.Path object

    # Check if the file exists
    if not weather_file_path.is_file():
        logger.info("Weather data not found in cache. Downloading new data.")
        download_url = _create_url_uur(begin_date, end_date, lat, lon)
        _download_csv_from_url(download_url, weather_file_path)

    weather_df = pd.read_csv(weather_file_path)
    return weather_df


def load_weather_data_cds(begin_date: datetime.date, end_date: datetime.date, lat=52.1, lon=5.18):
    # Check if weather folder exists
    if not downloaded_weather_cds_folder.is_dir():
        downloaded_weather_cds_folder.mkdir()

    # Format dates
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)

    # Create file name
    weather_file = f"Climate_Data_Store_{begin_date_str}_{end_date_str}.csv"
    weather_file_path = downloaded_weather_cds_folder / weather_file  # pathlib.Path object

    # Check if the file exists
    if not weather_file_path.is_file():
        logger.info("Weather data not found in cache. Downloading new data.")
        download_url = _create_url_cds(begin_date, end_date, lat, lon)
        _download_csv_from_url(download_url, weather_file_path)

    weather_df = pd.read_csv(weather_file_path)
    return weather_df


# the soil level can be changed but most import is level 3
def extract_soil_temperature(level=3):
    if level not in [1,2,3,4]:
        return "Only levels 1, 2, 3 and 4 exist"
    T_soil = weather_data["soil_temperature_level_{0}".format(level)]
    return T_soil



lat,lon = XY_to_latlon(circuit_1358)
time_bound = (pd.Timestamp(2019, 1, 29), pd.Timestamp(2021, 11, 30))
weather_data = load_weather_data_cds(time_bound[0], time_bound[1], lat, lon)

T_soil = extract_soil_temperature().to_numpy() # as numpy array