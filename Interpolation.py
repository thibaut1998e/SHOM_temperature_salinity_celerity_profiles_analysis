import numpy as np
import os
from utils import *
from plot_profile_locations import plot_data


def interpolation_map(train_long_lat, labels, test_long_lat, const=0, threshold=15,
                      sigma=15, title='', save_location=None, show=True):
    """Make predictions on points in test_long_lat
    plot the map with the predictions
    train_long_lat, test_long_lat:  2D array, shape (n_profiles, 2) train_long_lat[i][0] : longitudes, train_long_lat[i][1] : latitudes
    labels: associated labels (either temperature or salinity)
    const : constant to add to all the prediction, depends on the day of the year
    title : title of the map saved
    save_location, if None the map is not saved ,
    show : map is shown iff show

    """
    print(f'processing {len(test_long_lat)} test data')
    test_predictions = [gaussian_weight_interpolation(p, train_long_lat, labels, threshold, sigma) + const for p in test_long_lat]
    plot_data(test_long_lat.T[0], test_long_lat.T[1], test_predictions, show=False)  # plot real data
    labels = [label + const for label in labels]
    plot_data(train_long_lat.T[0], train_long_lat.T[1], labels, title=title, save_location=save_location,
                  show=show)  # plot results of interpolation


def interpolation_validation(train_long_lat, labels, valid_long_lat, valid_labels, valid_dates, average_temps,
                             threshold=15, sigma=None, lat_reg=None, densities=[]):
    """compute the average error of the interpolation on valid_long_lat"""
    print(f'processing {len(valid_long_lat)} validation data')
    global cpt
    cpt = 0
    # The following if/elif/else instructions prepare the list of standard deviations sigmas
    if sigma != None :
        sigmas = [sigma for i in range(len(valid_long_lat))]
    elif densities != [] :
        sigmas = []
        for i,p in enumerate(valid_long_lat) :
            density = density_profiles_nn_interpolation(densities, p)
            #print(density)
            #sigmas.append(np.exp(-np.log(1+np.log(1+density))) * .3)
            sigmas.append(np.sqrt(np.exp(-np.log((1+density)))))
        print(sigmas)
    else :
        sigmas = [None for i in range(len(valid_long_lat))]
    valid_predictions = [gaussian_weight_interpolation(p, train_long_lat, labels, threshold, sigmas[i])
                         + average_temps[valid_dates[i]-1] for i,p in enumerate(valid_long_lat)]
    # influence of the latitude, if needed
    if lat_reg != None :
        for i,p in enumerate(valid_long_lat) :
            valid_predictions[i] += lat_reg.predict(np.array([[p[1]]]))[0]
    
    error = distance(valid_predictions, valid_labels, order=1) / len(valid_predictions)
    
    
    #print([valid_predictions[i] - valid_labels[i] for i in range(len(valid_labels))])
    if sigma == None and densities != [] :
        print(f"Mean of sigmas : {mean(sigmas)}")
        print("Weighted sum of errors :")
        print(sum([abs(valid_predictions[i] - valid_labels[i]) * sigmas[i] for i in range(len(sigmas))]) / sum(sigmas))
        print("Unweighted sum of errors :")
        print(mean([abs(valid_predictions[i] - valid_labels[i]) for i in range(len(sigmas))]))
    return error

cpt = 0


def gaussian_weight_interpolation(p, X, Y, threshold, sigma):
    """p point with 2 coordinate : p[0] latitude, p[1] longitude
    X 2D array shape (N, d) N number of data point, d number of dimension (2 in our case longitude and latitude)
    Y 1D array, shape N, associated labels
    Threshold : only points closer than threshold are taken into acount in the interpolation
    sigm : std of the gaussian kernel
    returns the prdicted value for p (either temperature or salinity)"""
    global cpt
    cpt += 1
    if cpt % 10 == 0:
        print(cpt)
    closest_point_values = []
    closest_point_distances = []
    weights = []
    if sigma != None :
        for i,x in enumerate(X):
            dist = distance(p, x)
            if dist < threshold:
                w = gauss(dist, sigma)
                weights.append(w)
                closest_point_values.append(Y[i])
        closest_point_values = np.array(closest_point_values)
        S = sum(weights)
        weights = np.array([w / S for w in weights])
    else :
        # if argument sigma is None then we define sigma as the double of the distance to the closest point
        for i,x in enumerate(X):
            dist = distance(p, x)
            if dist < threshold:
                closest_point_values.append(Y[i])
                closest_point_distances.append(dist)
        min_dist = min(closest_point_distances)
        sigma = 2 * min_dist if min_dist > 0 else 0.001
        
        print(sigma)
        for dist in closest_point_distances :
            weights.append(gauss(dist, sigma))
        closest_point_values = np.array(closest_point_values)
        S = sum(weights)
        weights = np.array([w / S for w in weights])
    y_pred = closest_point_values.dot(weights.T) #predicted value for p
    return y_pred




if __name__ == '__main__':
    from temperature_residuals import surface_temps_2, surface_temps_3, lons, lats, smoothed_Y, valid_profs, reg
    #from utils import get_profiles, LIGHT_PROF

    day = 25
    lons = np.array(lons)
    lats = np.array(lats)
    surface_temps_2 = np.array(surface_temps_2)
    surface_temps_3 = np.array(surface_temps_3)
    #print('mean of residuals (without removing lat influence)', np.mean(surface_temps_2))
    #print('std of residuals (without removing lat influence)', np.std(surface_temps_2))
    smoothed_Y = np.array(smoothed_Y)
    long_lat = np.array([lons, lats]).T
    title = f'predicted surface temperatures in the Mediterranean Sea at day {day}'
    save_location = f'map_mediterane_temperatures_day_{day}.png'
    const = smoothed_Y[day-1]
    valid_lon = get_longs(valid_profs)
    valid_lat = get_lats(valid_profs)
    valid_temps = get_surf_temp(valid_profs)
    valid_dates = get_date(valid_profs)
    long_lat_valid = np.array([valid_lon, valid_lat]).T
    long_lats, densities = read_density_txt() #get the saved values of densities
    import time
    t = time.time()
    #interpolation_map(long_lat, surface_temps_2, grid, const=const, title=title, save_location=save_location, show=True)
    #0.25s for one point on average
    """
    error = interpolation_validation(long_lat, surface_temps_2, long_lat_valid, valid_temps, valid_dates, smoothed_Y)
    print(error)
    print('temps écoulé', time.time() - t)
    """
    #errors_against_sigma_2 = [interpolation_validation(long_lat, surface_temps_2, long_lat_valid, valid_temps, valid_dates
    #                                                 ,smoothed_Y, sigma=None, threshold=3) for sig in [1]]
    errors_against_sigma_3 = [interpolation_validation(long_lat, surface_temps_3, long_lat_valid, valid_temps, valid_dates
                                                     ,smoothed_Y, sigma=.1, threshold=5, lat_reg=reg) for sig in [1]]
    errors_against_sigma_4 = [interpolation_validation(long_lat, surface_temps_3, long_lat_valid, valid_temps, valid_dates
                                                     ,smoothed_Y, sigma=None, threshold=5, lat_reg=reg, densities=[]) for sig in [1]]
    # errors_against_sigma_3 uses a constant sigma of value 0.1
    # errors_against_sigma_4 uses a sigma that depends on the distance to the closest point
    print(errors_against_sigma_3, errors_against_sigma_4)
















