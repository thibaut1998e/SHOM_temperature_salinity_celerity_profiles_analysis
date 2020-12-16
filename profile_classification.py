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
from sklearn.mixture import GaussianMixture

def train_and_return_GMM_model(profiles, n_groups, **args_gaussian_mixture):
    GMM = GaussianMixture(n_groups, **args_gaussian_mixture)
    X = np.array([prof.celerities for prof in profiles])
    GMM.fit(X)
    return GMM

def GMM_classification(profiles, model):
    X = np.array([prof.celerities for prof in profiles])
    return model.predict(X)

def train_and_return_kmeans_model(profiles, n_groups):
    kmeans = KMeans(n_groups, init='k-means++', n_init=1)
    X = np.array([prof.celerities for prof in profiles])
    kmeans.fit(X)
    return kmeans





def k_means_classification(profiles, n_groups, model=None):
    X = np.array([prof.celerities for prof in profiles])
    if model != None:
        return model.predict(X)
    else:
        kmeans = KMeans(n_groups, init='k-means++', n_init=1)
        return kmeans.fit_predict(X)


def dbscan_classification(profiles, epsilon, min_samples_in_group):
    X = np.array([prof.celerities for prof in profiles])
    return dbscan(X, epsilon, min_samples_in_group)


def plot_classification(profiles, classification, **optional_arg_plot_data):
    longs = [profile.prof.lon for profile in profiles]
    lats = [profile.prof.lat for profile in profiles]
    plot_data(longs, lats, classification, **optional_arg_plot_data)


def get_mean_profiles(profiles, classification):
    nb_groups = max(classification) + 1
    mean_profiles = []
    for group in range(nb_groups):
        nb_profiles = 0
        group_profiles = np.zeros(len(profiles[0].celerities))
        for ind in range(len(classification)):
            if classification[ind] == group:
                group_profiles += profiles[ind].celerities
                nb_profiles += 1
        mean_profiles.append(group_profiles / nb_profiles)
    return np.array(mean_profiles)






if __name__ == '__main__':
    domain = [50 * i for i in range(1, 11)]
    print(f"The domain goes from depth {domain[0]} m to {domain[-1]} m.")
    PROFS = get_all_profiles()
    projected_PROFS = get_projected_celerity_profiles(PROFS, domain)
    print(f"The number of profiles that can be projected on the domain is {len(projected_PROFS)}.")

    suppress_nans_celerities(projected_PROFS)

    def test_kmean():
        nb_classes_kmeans = 4
        kmeans_model = train_and_return_kmeans_model(projected_PROFS, nb_classes_kmeans)
        centers_groups = kmeans_model.cluster_centers_
        plot_profiles(centers_groups, xlabel='celerity', title=f'center of kMean classification with {nb_classes_kmeans} classes')

        classif_kmeans = k_means_classification(projected_PROFS, nb_classes_kmeans)
        plot_classification(projected_PROFS, classif_kmeans, title=f'Kmean classification with {nb_classes_kmeans} classes')
        X = np.array([prof.celerities for prof in projected_PROFS])
        """
        silhouettes = []
        ks = range(2, 10)
        for k in ks:
            kmeans = train_and_return_kmeans_model(projected_PROFS, k)
            silhouette = silhouette_score(X, kmeans.labels_)
            silhouettes.append(silhouette)
        plt.plot(ks, silhouettes)
        plt.title('kMean, silhouette factor wrt number of classes')
        plt.show()
        """
    def test_db_SCAN():
        classif_dbscan = dbscan_classification(projected_PROFS, 0.9, 100)[1]
        classified_PROFS_idx = []
        for ind in range(len(projected_PROFS)):
            if classif_dbscan[ind] != -1:
                classified_PROFS_idx.append(ind)
        classified_PROFS = [projected_PROFS[ind] for ind in classified_PROFS_idx]
        classif_dbscan = [classif_dbscan[ind] for ind in classified_PROFS_idx]
        proportion_classified = len(classified_PROFS_idx) / len(projected_PROFS)
        print(f"The number of profiles classified by dbscan is {len(classified_PROFS_idx)}.")
        print(f"This corresponds to {100 * round(proportion_classified, 3)}% of the projected profiles.")

        #plot_classification([projected_PROFS[ind] for ind in classified_PROFS_idx],
                            #[classif_dbscan[ind] for ind in classified_PROFS_idx])

        centers_dbscan = get_mean_profiles(classified_PROFS, classif_dbscan)
        plot_profiles(centers_dbscan, title='center of dbScan classification', xlabel='celerity')


    def test_GMM():
        for cov_type in ['full', 'diag', 'tied', 'spherical']:
            nb_classes_GMM = 4
            GMM_model = train_and_return_GMM_model(projected_PROFS, nb_classes_GMM, covariance_type=cov_type)
            classif_GMM = GMM_classification(projected_PROFS, GMM_model)
            plot_classification(projected_PROFS, classif_GMM,
                                title=f'Gaussian mixture model classification with {nb_classes_GMM} classes, covariance type : {cov_type}')
            centers = get_mean_profiles(projected_PROFS, classif_GMM)
            plot_profiles(centers, axs_shape=(1, nb_classes_GMM), title=f'center of GMM covariance type {cov_type}')




    # test_kmean()
    #test_db_SCAN()
    test_GMM()






'''
surf = []
temp = []
for prof in PROFS :
    surf.append(prof.depth[-1])
    temp.append(prof.temp[-1])
plt.scatter(surf, temp)
'''





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
