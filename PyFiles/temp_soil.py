#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyproj import Transformer

# Circuits coordinates
CIRCUIT_COORDINATES =  {1358:(131436.240,502678.470),
                        2003:(117955.218,479433.147),
                        2308:(105884.568,465322.411),
                        2611:(111530.098,516758.349),
                        2821:(114053.548,471692.944),
                        3249:(163329.228,564016.874),
                        3410:(132660.045,501028.933),
                        3512:(198110.621,585155.202),
                        3543:(106754.478,500162.533),
                        20049:(165236.264,577272.523),
                        20726:(206863.49,567024.869),
                        22102:(158361.29,432857.689)}

from weather_api import *


def load_temp_soil(circuitnr, begin_date, end_date, level=3):
    """
    Load the soil temperature data (Celcius) from online weather API
    :param circuitnr: circuitnr which to load
    :param level: depth level of soil temperature (1,2,3,4)
    :return dataframe with soil temperature (Kelvin) with hourly timestamps as index
    """
    circuitnr = int(circuitnr)

    #check whether the circuit number eists
    if circuitnr not in CIRCUIT_COORDINATES:
        return "circuit number doesn't exist"

    # Check for correct level input
    if level not in [1,2,3,4]:
        return "Only levels 1, 2, 3 and 4 exist"

    # Transforms from Dutch Coordinate System (Amersfoort / RD New) to World Coordinate System (WGS84)
    transformer = Transformer.from_crs( "EPSG:28992","EPSG:4326")
    lat, lon = transformer.transform(CIRCUIT_COORDINATES[circuitnr][0],CIRCUIT_COORDINATES[circuitnr][1])

    weather_data = load_weather_data_cds(lat, lon, circuitnr)

    # Extract the desired time interval
    temp_soil = weather_data["soil_temperature_level_{0}".format(level)][begin_date:end_date]

    # Change from Celcius to Kelvin
    temp_soil += 273.15

    return temp_soil