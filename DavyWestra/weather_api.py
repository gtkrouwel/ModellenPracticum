#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import os
from pathlib import Path

import pandas as pd
import requests

path_to_alliander_repo = Path(os.pardir, os.pardir, "modellenpracticum2022-speed-of-heat")
downloaded_weather_cds_folder = path_to_alliander_repo / "data" / "weather_cds_data"

logger = logging.getLogger(__name__)

def _format_date(unformatted_date: datetime.date):
    return unformatted_date.strftime("%Y-%m-%d")


def _create_url(begin_date: datetime.date, end_date: datetime.date, lat=52.1, lon=5.18):
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)
    url_template = f"https://weather.appx.cloud/api/v2/weather/sources/knmi/models/daggegevens?begin={begin_date_str}&end={end_date_str}&lat={lat}&lon={lon}&units=human&response_format=csv"
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


def load_weather_data_cds(begin_date: datetime.date, end_date: datetime.date, lat, lon, circuitnr):
    # Check if weather folder exists
    if not downloaded_weather_cds_folder.is_dir():
        downloaded_weather_cds_folder.mkdir()

    # Format dates
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)

    # Create file name
    weather_file = f"Climate_Data_Store_{begin_date_str}_{end_date_str}_{circuitnr}.csv"
    weather_file_path = downloaded_weather_cds_folder / weather_file  # pathlib.Path object

    # Check if the file exists
    if not weather_file_path.is_file():
        logger.info("Weather data not found in cache. Downloading new data.")
        download_url = _create_url_cds(begin_date, end_date, lat, lon)
        _download_csv_from_url(download_url, weather_file_path)

    weather_df = pd.read_csv(weather_file_path)
    return weather_df