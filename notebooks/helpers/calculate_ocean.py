import numpy as np
from global_land_mask import globe


def is_ocean(latitude, longitude):
    return globe.is_ocean(latitude, longitude)
