#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyproj import Transformer
import datetime
import os
from pathlib import Path

import pandas as pd
import requests

# Circuits coordinates
circuit_coordinates =  {1358:(131436.240,502678.470),
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

# For begin_date and end_date you can use pd.Timestamp(YYYY,MM,DD)
def T_soil(circuitnr, begin_date: datetime.date, end_date: datetime.date,level=3):
    circuitnr = int(circuitnr)

    #check whether the circuit number eists
    if circuitnr not in circuit_coordinates:
        return "circuit number doesn't exist"
    else:
        circuitnr_XY = circuit_coordinates.get(circuitnr)

    # Check for correct level input
    if level not in [1,2,3,4]:
        return "Only levels 1, 2, 3 and 4 exist"

    # Transforms from Dutch Coordinate System (Amersfoort / RD New) to World Coordinate System (WGS84)
    transformer = Transformer.from_crs( "EPSG:28992","EPSG:4326")
    lat, lon = transformer.transform(circuit_coordinates[circuitnr][0],circuit_coordinates[circuitnr][1])

    weather_data = load_weather_data_cds(lat, lon, circuitnr)

    # Extract the desired time interval
    T_soil = weather_data["soil_temperature_level_{0}".format(level)][begin_date:end_date]
    return T_soil