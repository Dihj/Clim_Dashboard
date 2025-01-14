# Function for my dashboard
# load library
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from matplotlib import cm, colors
import scipy.stats as stats
from scipy.stats import t, norm
import yaml
import copy

#####################################################################
# Copyright tfeldmann, MIT license.
# https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
def deep_merge(a: dict, b: dict) -> dict:
    result = copy.deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = deep_merge(av, bv)
        else:
            result[bk] = copy.deepcopy(bv)
    return result


def load_config(colon_separated_filenames):
    filenames = colon_separated_filenames.split(":")
    config = {}
    for fname in filenames:
        with open(fname) as f:
            config = deep_merge(config, yaml.safe_load(f))
    return config
######################################################################

# Load NetCDF data and calculate monthly climatology
def load_netcdf_data(filepath):
    dataset = Dataset(filepath, mode='r')
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    time = dataset.variables['time'][:]
    rainfall = dataset.variables['precip'][:]
    
    # Convert time to a pandas datetime index
    time_index = pd.to_datetime("1981-01-01") + pd.to_timedelta(time, unit='D')
    
    # Create a DataFrame to store rainfall data with time and coordinate indices
    rainfall_df = pd.DataFrame(data=rainfall.reshape((rainfall.shape[0], -1)), index=time_index)
    
    # Monthly climatology
    monthly_climatology = rainfall_df.groupby(rainfall_df.index.month).mean()
    
    return latitudes, longitudes, monthly_climatology, rainfall_df


# Function to get color based on rainfall value
def get_color(value, max_value):
    if np.isnan(value):
        return 'transparent'
    norm_value = value / max_value
    color = cm.get_cmap('YlGnBu')(norm_value)
    return f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})'
