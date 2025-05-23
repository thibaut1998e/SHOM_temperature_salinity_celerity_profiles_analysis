import pickle
import matplotlib.pyplot as plt
from utils import LIGHT_PROF, get_profiles


PROFS = get_profiles()
lats = [prof.lat for prof in PROFS]
lons = [prof.lon for prof in PROFS]

plt.scatter(lons, lats, s=0.5)

import os

# Chemin à remplacer par l'emplacement du fichier epsg pour vous
os.environ['PROJ_LIB'] = "C:/Users/Thibaut/anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share"

# Pour installer basemap avec Anaconda : conda install -c anaconda basemap
from mpl_toolkits.basemap import Basemap

map = Basemap(llcrnrlon=-10, llcrnrlat=25, urcrnrlon=40, urcrnrlat=50)
map.drawcoastlines(linewidth=0.5)

# plt.show()

from scipy.spatial import Delaunay
import numpy as np

n = len(PROFS)
points = np.zeros((n, 2))
for i in range(n):
    points[i] = np.array([lons[i], lats[i]])

tri = Delaunay(points)
plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())

plt.show()