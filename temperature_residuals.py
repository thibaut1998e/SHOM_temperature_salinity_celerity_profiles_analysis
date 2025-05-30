
import pickle
import matplotlib.pyplot as plt
from utils import *


plot = (__name__ == '__main__')

# load profiles

_, PROFS, valid_profs = get_profiles()

lats = get_lats(PROFS)
lons = get_longs(PROFS)
surface_temps = get_surf_temp(PROFS)

#dates = [prof.date for prof in PROFS]
integer_dates = get_date(PROFS)

# Removing a few disturbing data
good_data = []
for i in range(len(PROFS)) :
    if integer_dates[i] < 258 or integer_dates[i] > 263 or surface_temps[i] > 18 :
        good_data.append(PROFS[i])

PROFS = good_data
PROFS.sort(key = lambda prof: transform_string_date_into_integer(str(prof.date)))
lats = get_lats(PROFS)
lons = get_longs(PROFS)
surface_temps = get_surf_temp(PROFS)
integer_dates = get_date(PROFS)


# Plots all the surface temperatures recorded on the time axis
if plot:
    plt.scatter(integer_dates, surface_temps)
    plt.title('surface temperatures recorded on the time axis')
    plt.xlabel('time (idx of days)')
    plt.ylabel('surface temperature (°C)')
    plt.show()




n = len(PROFS)
i = 1
mean_surface_temps = {}
sum_of_surface_temps = surface_temps[0]
number_of_profiles = 1
while i < n :
    if integer_dates[i] == integer_dates[i-1] :
        if integer_dates[i] < 258 or integer_dates[i] > 263 or surface_temps[i] > 18 :
            sum_of_surface_temps += surface_temps[i]
            number_of_profiles += 1
    else :
        mean_surface_temps[integer_dates[i-1]] = sum_of_surface_temps / number_of_profiles
        sum_of_surface_temps = surface_temps[i]
        number_of_profiles = 1
    i += 1
if integer_dates[-1] not in mean_surface_temps :
    mean_surface_temps[integer_dates[-1]] = sum_of_surface_temps / number_of_profiles

# Looks for the missing days and fills them with the surrounding information
X = list(mean_surface_temps)
missing_ranges = []
i = 1
while i <= 364 :
    if i not in X :
        j = i
        while j not in X :
            j += 1
        missing_ranges.append((i, j))
        i = j
    else :
        i += 1
for (i, j) in missing_ranges :
    for k in range(i, j) :
        diff = mean_surface_temps[j] - mean_surface_temps[i-1]
        mean_surface_temps[k] = mean_surface_temps[i-1] + diff * (k-i+1) / (j-i+1)

X = [i for i in range(1, 366)]
Y = [mean_surface_temps[x] for x in X]



# Appying a uniform kernel to make the Y data smoother
number_days = 9
smoothed_Y = []
for d in range(number_days // 2) :
    smoothed_Y.append(Y[d])
for d in range(number_days // 2, len(Y) - number_days // 2) :
    smoothed_Y.append(sum(Y[d - number_days // 2 : d + number_days // 2 + 1]) / number_days)
for d in range(number_days // 2) :
    smoothed_Y.append(Y[len(Y) - number_days // 2 + d])


if plot:
    # Plots the means of the recorded surface temperatures over time
    plt.plot(X, Y, color='red', label='real mean')
    # Plots the smoothed average of surface temperature over time
    plt.plot(X, smoothed_Y, color='orange', label='smoothed mean')
    plt.title('mean of recorded surface temperature over time')
    plt.xlabel('time (day idx)')
    plt.ylabel('T (°C)')
    plt.legend()
    plt.show()

# Creates residuals by removing the estimated influence of the day to the raw surface temperature data
surface_temps_2 = []
for ind in range(len(PROFS)) :
    prof = PROFS[ind]
    surface_temps_2.append(prof.temp[-1] - smoothed_Y[integer_dates[ind]-1])

# Plots the residuals after removing the influence of the day
if plot:
    plt.scatter(integer_dates, surface_temps_2)
    plt.title('residuals after removing the influence of the day')
    plt.xlabel('time (day idx)')
    plt.ylabel('Temperature residuals (°C)')
    plt.show()




import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array(lats).reshape(-1, 1)
y = np.array(surface_temps_2)
reg = LinearRegression().fit(X, y)
# Plots the temperature residuals against their latitudes

if plot:
    plt.scatter(lats, surface_temps_2)
    plt.plot(X, reg.predict(X), color='red', label='fitted linear regression model')
    plt.title('temperature residuals (after removing the influence of the day) against their latitudes')
    plt.xlabel('latitude')
    plt.ylabel('Temperature residuals (°C)')
    plt.show()

    plt.scatter(lons, surface_temps_2)
    plt.title('temperature residuals (after removing the influence of the day) against their longitude')
    plt.xlabel('longitude')
    plt.ylabel('Temperature residuals (°C)')
    plt.show()


# Creates residuals by removing the estimated influence of the latitude from the previous residuals
surface_temps_3 = []
for ind in range(len(PROFS)) :
    surface_temps_3.append(surface_temps_2[ind] - reg.predict(np.array([[lats[ind]]]))[0])

if plot:
    # Plots the residual surface temperatures after removing the influences of the day and of the latitude
    plt.scatter(integer_dates, surface_temps_3)
    plt.title('residual surface temperatures after removing the influences of the day and of the latitude')
    plt.xlabel('time (day idx)')
    plt.ylabel('Temperature residuals (°C)')
    plt.show()

    plt.scatter(lons, surface_temps_3)
    plt.title('temperature residuals after removing the influence of the day and the latitude against their longitude')
    plt.xlabel('longitude')
    plt.ylabel('Temperature residuals (°C)')
    plt.show()


surface_temp_residuals = surface_temps_3


print(f"Mean of surface temperature residuals : {mean(surface_temp_residuals)}")
print(f"Variance of surface temperature residuals : {var(surface_temp_residuals)}")


#plt.show()