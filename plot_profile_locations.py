# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:04:39 2020

@author: Anthony
"""
from utils import LIGHT_PROF, get_profiles
import pickle
import matplotlib.pyplot as plt
from path import path_base_map


# profiles class
def plot_data(longs, lats, labels, show=True, title='', save_location=None):
    """In: longs, lats : longitudes and latitudes 1D array (or list) shape (nb_profiles)
     labels : 1D array shape (or list) (nb_profile) associated labels (either temperature or salinity)"""
    import os
    os.environ['PROJ_LIB'] = path_base_map #change this variable to your corresponding path in file path.py
    from mpl_toolkits.basemap import Basemap
    plt.scatter(longs, lats, s=0.5, c=labels)
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
    PROFS = get_profiles()
    plot_profiles_surface_temp(PROFS)



