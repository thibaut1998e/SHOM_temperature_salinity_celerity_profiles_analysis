import numpy as np
import os
from utils import *
from plot_profile_locations import plot_data
import matplotlib.pyplot as plt





def interpolation_map(data_long_lat, labels, const=0, lat_min=32, lat_max=44, lat_step=1, long_min=-4, long_max=35,
                      long_step=1, threshold=15, sigma=15, title='', save_location=None):
    """data_long_lat 2D array, shape (n_profiles, 2) data_long_lat[i][0] : longitudes, data_long_lat[i][1] : latitudes
    labels : associated labels (either temperature or salinity)
    const : constant to add to all the prediction """
    longitudes = np.arange(long_min, long_max, long_step)
    latitudes = np.arange(lat_min, lat_max, lat_step)
    grid = cartesian_product(longitudes, latitudes)
    print('len grid', len(grid))
    values = np.zeros(len(grid))
    for i, point in enumerate(grid):
        print(i)
        values[i] = gaussian_weight_interpolation(point, data_long_lat, labels, threshold, sigma) + const
        print(values[i])
    plot_data(grid.T[0], grid.T[1], values, show=False) # plot real data
    labels = [label + const for label in labels]
    plot_data(data_long_lat.T[0], data_long_lat.T[1], labels, title=title, save_location=save_location) #plot results of interpolation


def gaussian_weight_interpolation(p, X, Y, threshold, sigma):
    """p point with 2 coordinate : p[0] latitude, p[1] longitude
    X 2D array shape (N, d) N number of data point, d number of dimension (2 in our case longitude and latitude)
    Y 1D array, shape N, associated labels
    Threshold : only points closer than threshold are taken into acount in the interpolation
    sigm : std of the gaussian kernel
    returns the prdicted value for p (either temperature or salinity)"""
    closest_point_values = []
    weights = []
    for i,x in enumerate(X):
        dist = distance(p, x)
        if dist < threshold:
            w = gauss(dist, sigma)
            weights.append(w)
            closest_point_values.append(Y[i])
    closest_point_values = np.array(closest_point_values)
    S = sum(weights)
    weights = np.array([w / S for w in weights])
    y_pred = closest_point_values.dot(weights.T) #predicted value for p
    return y_pred


if __name__ == '__main__':
    from temperature_residuals import surface_temps_2, lons, lats, smoothed_Y
    from utils import get_profiles, LIGHT_PROF
    day = 25
    lons = np.array(lons)
    lats = np.array(lats)
    surface_temps_2 = np.array(surface_temps_2)
    smoothed_Y = np.array(smoothed_Y)
    long_lat = np.array([lons, lats]).T
    title = f'predicted surface temperatures in the Mediterranean Sea at day {day}'
    save_location = f'map_mediterane_temperatures_day_{day}.png'
    const = smoothed_Y[day-1]
    interpolation_map(long_lat, surface_temps_2, const=const, title=title, save_location=save_location)













