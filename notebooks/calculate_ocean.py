import numpy as np
from global_land_mask import globe
from geopy.distance import great_circle

def is_ocean(latitude, longitude):
    return globe.is_ocean(latitude, longitude)