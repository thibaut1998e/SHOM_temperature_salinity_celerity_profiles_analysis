# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:04:39 2020

@author: Anthony
"""
from plot_data import LIGHT_PROF, get_profiles
import pickle
import matplotlib.pyplot as plt


# profiles class


if __name__ == '__main__':
    PROFS = get_profiles()
    lats = [prof.lat for prof in PROFS]
    lons = [prof.lon for prof in PROFS]


    surface_temps = [prof.temp[-1] for prof in PROFS]

    plt.scatter(lons, lats, s=0.5, c=surface_temps)
    plt.colorbar()

    import os

    # Chemin à remplacer par l'emplacement du fichier epsg pour vous
    os.environ['PROJ_LIB'] = "C:/Users\\Anthony\Anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share"

    # Pour installer basemap avec Anaconda : conda install -c anaconda basemap
    from mpl_toolkits.basemap import Basemap

    map = Basemap(llcrnrlon=-10, llcrnrlat=25, urcrnrlon=40, urcrnrlat=50)
    map.drawcoastlines(linewidth=0.5)

    print("Plotting all the profile locations with their associated surface temperatures.")

    plt.show()