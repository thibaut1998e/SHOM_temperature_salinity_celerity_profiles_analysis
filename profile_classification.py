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
from sklearn.cluster import dbscan
from sklearn.metrics import silhouette_score



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

def train_and_return_kmeans_model(profiles, n_groups) :
    kmeans = KMeans(n_groups, init='k-means++', n_init=1)
    X = np.array([prof.celerities for prof in profiles])
    kmeans.fit(X)
    return kmeans

def k_means_classification(profiles, n_groups, model=None) :
    
    X = np.array([prof.celerities for prof in profiles])
    if model != None :
        return model.predict(X)
    else :
        kmeans = KMeans(n_groups, init='k-means++', n_init=1)
        return kmeans.fit_predict(X)

def dbscan_classification(profiles, epsilon, min_samples_in_group) :
    X = np.array([prof.celerities for prof in profiles])
    return dbscan(X, epsilon, min_samples_in_group)


def plot_classification(profiles, classification) :
    longs = [profile.prof.lon for profile in profiles]
    lats = [profile.prof.lat for profile in profiles]
    plot_data(longs, lats, classification)
   
def get_mean_profiles(profiles, classification) :
    nb_groups = max(classification)+1
    mean_profiles = []
    for group in range(nb_groups) :
        nb_profiles = 0
        group_profiles = np.zeros(len(profiles[0].celerities))
        for ind in range(len(classification)) :
            if classification[ind] == group :
                group_profiles += profiles[ind].celerities
                nb_profiles += 1
        mean_profiles.append(group_profiles / nb_profiles)
    return mean_profiles

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

#classif_kmeans = k_means_classification(projected_PROFS, 3)
#plot_classification(projected_PROFS, classif_kmeans)

'''
classif_dbscan = dbscan_classification(projected_PROFS, 0.9, 100)[1]
classified_PROFS_idx = []
for ind in range(len(projected_PROFS)) :
    if classif_dbscan[ind] != -1 :
        classified_PROFS_idx.append(ind)
classified_PROFS = [projected_PROFS[ind] for ind in classified_PROFS_idx]
classif_dbscan = [classif_dbscan[ind] for ind in classified_PROFS_idx]
proportion_classified = len(classified_PROFS_idx) / len(projected_PROFS)
print(f"The number of profiles classified by dbscan is {len(classified_PROFS_idx)}.")
print(f"This corresponds to {100 * round(proportion_classified, 3)}% of the projected profiles.")

#plot_classification([projected_PROFS[ind] for ind in classified_PROFS_idx], 
#                    [classif_dbscan[ind] for ind in classified_PROFS_idx])

centers_dbscan = get_mean_profiles(classified_PROFS, classif_dbscan)
'''

kmeans_model_3 = train_and_return_kmeans_model(projected_PROFS, 3)
centers_3_groups = kmeans_model_3.cluster_centers_
    
kmeans_model_4 = train_and_return_kmeans_model(projected_PROFS, 4)
centers_4_groups = kmeans_model_4.cluster_centers_
    
'''
X = np.array([prof.celerities for prof in projected_PROFS])
for k in range(2, 10) :
    kmeans = train_and_return_kmeans_model(projected_PROFS, k)
    silhouette = silhouette_score(X, kmeans.labels_)
    print(k, silhouette)
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
