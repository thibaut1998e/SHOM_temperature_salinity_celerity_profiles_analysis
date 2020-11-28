# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:30:48 2020

@author: Anthony
"""

from utils import *
from plot_profile_locations import *

import copy as cp
import matplotlib.pyplot as plt
from math import isnan

from sklearn.cluster import KMeans



def suppress_nans(profiles) :
    for prof in profiles :
        celerities = prof.celerities
        for ind in range(len(celerities)) :
            if isnan(celerities[ind]) :
                first_ind = ind
                last_ind = ind+1 if ind < len(celerities)-1 else ind
                while last_ind < len(celerities)-1 and isnan(celerities[last_ind]) :
                    last_ind += 1
                if first_ind == 0 :
                    for i in range(first_ind, last_ind+1) :
                        celerities[i] = celerities[last_ind+1]
                elif last_ind == len(celerities)-1 :
                    for i in range(first_ind, last_ind+1) :
                        celerities[i] = celerities[first_ind-1]
                else :
                    for i in range(first_ind, last_ind+1) :
                        celerities[i] = (celerities[first_ind-1] + celerities[last_ind+1]) / 2



def k_means_classification(profiles, n_groups) :
    kmeans = KMeans(n_groups, init='k-means++', n_init=1)
    X = np.array([prof.celerities for prof in profiles])
    return kmeans.fit_predict(X)




def plot_classification(profiles, classification) :
    longs = [profile.prof.lon for profile in profiles]
    lats = [profile.prof.lat for profile in profiles]
    plot_data(longs, lats, classification)
   

domain = [50*i for i in range(1, 11)]
print(f"The domain goes from depth {domain[0]} m to {domain[-1]} m.")
PROFS = get_all_profiles()
'''
surf = []
temp = []
for prof in PROFS :
    surf.append(prof.depth[-1])
    temp.append(prof.temp[-1])
plt.scatter(surf, temp)
'''
projected_PROFS = get_projected_celerity_profiles(PROFS, domain)
print(f"The number of profiles that can be projected on the domain is {len(projected_PROFS)}.")

suppress_nans(projected_PROFS)

classif = k_means_classification(projected_PROFS, 4)
plot_classification(projected_PROFS, classif)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    