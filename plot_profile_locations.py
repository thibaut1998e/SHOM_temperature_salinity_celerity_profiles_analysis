# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:04:39 2020

@author: Anthony
"""
from utils import LIGHT_PROF, get_profiles
import pickle
import matplotlib.pyplot as plt
from path import path_base_map
import numpy as np



# profiles class
def plot_data(longs, lats, labels, show=True, title='', save_location=None, color_log_scale=False):
    """In: longs, lats : longitudes and latitudes 1D array (or list) shape (nb_profiles)
     labels : 1D array shape (or list) (nb_profile) associated labels (either temperature or salinity)"""
    import os
    os.environ['PROJ_LIB'] = path_base_map #change this variable to your corresponding path in file path.py
    from mpl_toolkits.basemap import Basemap
    if color_log_scale:
        labels = np.log(labels)
    plt.scatter(longs, lats, s=0.5, c=labels)
    plt.xlabel('longitudes')
    plt.ylabel('latitudes')
    plt.xticks(np.arange(-10, 40, 1)) #graduation of axes do not work
    plt.yticks(np.arange(25, 50, 1))
    plt.grid()
    plt.colorbar()
    plt.title(title)
    map = Basemap(llcrnrlon=-10, llcrnrlat=25, urcrnrlon=40, urcrnrlat=50)
    map.drawcoastlines(linewidth=0.5)
    if save_location is not None:
        plt.savefig(save_location)
    if show:
        plt.show()


def plot_profiles_surface_temp(PROFS):

    lats = [prof.lat for prof in PROFS]
    longs = [prof.lon for prof in PROFS]
    surface_temps = [prof.temp[-1] for prof in PROFS]
    plot_data(longs, lats, surface_temps)


if __name__ == '__main__':
    PROFS, _, _ = get_profiles()
    plot_profiles_surface_temp(PROFS)



