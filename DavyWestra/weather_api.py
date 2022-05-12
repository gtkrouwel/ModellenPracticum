#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
from pathlib import Path

import pandas as pd
import requests

downloaded_weather_cds_folder = Path(os.pardir, "weather_data")

import sys
sys.path.append(str(downloaded_weather_cds_folder.resolve()))

def _format_date(unformatted_date: datetime.date):
    return unformatted_date.strftime("%Y-%m-%d")


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


def load_weather_data_cds(lat, lon, circuitnr, begin_date = pd.Timestamp(2019, 5, 1), end_date = pd.Timestamp(2022, 2, 21), ):
    # Check if weather folder exists
    if not downloaded_weather_cds_folder.is_dir():
        downloaded_weather_cds_folder.mkdir()

    # Format dates
    begin_date_str = _format_date(begin_date)
    end_date_str = _format_date(end_date)

    # Create file name
    weather_file = f"Climate_Data_Store_{circuitnr}.csv"
    weather_file_path = downloaded_weather_cds_folder / weather_file  # pathlib.Path object

    # Check if the file exists
    if not weather_file_path.is_file():
        download_url = _create_url_cds(begin_date, end_date, lat, lon)
        _download_csv_from_url(download_url, weather_file_path)

    weather_data = pd.read_csv(weather_file_path)

    # Set 'time' column as datetime object and set it as index
    weather_data["time"] = pd.to_datetime(weather_data["time"])
    weather_data = weather_data.set_index("time")

    return weather_data