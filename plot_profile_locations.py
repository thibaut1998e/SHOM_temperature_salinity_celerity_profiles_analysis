# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:04:39 2020

@author: Anthony
"""

import pickle
import matplotlib.pyplot as plt


# profiles class
class LIGHT_PROF():
    def __init__(self, temp, salt, depth, lon, lat, date):
        self.salt = salt
        self.temp = temp
        self.depth = depth
        self.lon = lon
        self.lat = lat
        self.date = date
        return


# load profiles

with open('ts_profiles.pkl', 'rb') as f:
    PROFS = pickle.load(f)

lats = [prof.lat for prof in PROFS]
lons = [prof.lon for prof in PROFS]

plt.scatter(lons, lats, s=0.5)

import os

# Chemin à remplacer par l'emplacement du fichier epsg pour vous
os.environ['PROJ_LIB'] = "C:/Users\\Anthony\Anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share"

# Pour installer basemap avec Anaconda : conda install -c anaconda basemap
from mpl_toolkits.basemap import Basemap

map = Basemap(llcrnrlon=-10, llcrnrlat=25, urcrnrlon=40, urcrnrlat=50)
map.drawcoastlines(linewidth=0.5)

plt.show()