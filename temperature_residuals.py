# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:52:16 2020

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

surface_temps = [prof.temp[-1] for prof in PROFS]


def transform_string_date_into_integer(date):
    """IN : string in the form AAAA-MM-JJ 00:00:00 representing a date
    OUT : integer, day 1 is 2013-01-01"""
    year, month, day = date.split('-')
    day, _ = day.split(' ')
    year, month, day = int(year), int(month), int(day)
    lenght_of_months = [31,28,31,30,31,30,31,31,30,31,30,31]
    return 365*(year-2013) + sum(lenght_of_months[:month-1]) + day

def day_number_to_period_of_year(day) :
    while day > 365.25 :
        day -= 365.25
    return day

dates = [prof.date for prof in PROFS]
integer_dates = [transform_string_date_into_integer(str(date)) for date in dates]

good_data = []
for i in range(len(PROFS)) :
    if integer_dates[i] < 258 or integer_dates[i] > 263 or surface_temps[i] > 18 :
        good_data.append(PROFS[i])

PROFS = good_data
lats = [prof.lat for prof in PROFS]
lons = [prof.lon for prof in PROFS]
surface_temps = [prof.temp[-1] for prof in PROFS]
dates = [prof.date for prof in PROFS]
integer_dates = [transform_string_date_into_integer(str(date)) for date in dates]


# Plots all the surface temperatures recorded on the time axis
#plt.scatter(integer_dates, surface_temps)

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
'''
for ind in range(len(integer_dates)) :
    if surface_temps[i]
'''
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
# Plots the means of the recorded surface temperatures over time
#plt.plot(X, Y, color='red')
number_days = 9
smoothed_Y = []
for d in range(number_days // 2) :
    smoothed_Y.append(Y[d])
for d in range(number_days // 2, len(Y) - number_days // 2) :
    smoothed_Y.append(sum(Y[d - number_days // 2 : d + number_days // 2 + 1]) / number_days)
for d in range(number_days // 2) :
    smoothed_Y.append(Y[len(Y) - number_days // 2 + d])

# Plots the smoothed average of surface temperature over time
#plt.plot(X, smoothed_Y, color='orange')

surface_temps_2 = []
for ind in range(len(PROFS)) :
    prof = PROFS[ind]
    surface_temps_2.append(prof.temp[-1] - smoothed_Y[integer_dates[ind]-1])

# Plots the residuals after removing the influence of the day
plt.scatter(integer_dates, surface_temps_2)

# Plots the temperature residuals against their latitudes
#plt.scatter(lats, surface_temps_2)
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array(lats).reshape(-1, 1)
y = np.array(surface_temps_2)
reg = LinearRegression().fit(X, y)

#plt.plot(X, reg.predict(X))

surface_temps_3 = []
for ind in range(len(PROFS)) :
    surface_temps_3.append(surface_temps_2[ind] - reg.predict(np.array([[lats[ind]]]))[0])

# Plots the residuals after removing the influences of the day
# and of the latitude
plt.scatter(integer_dates, surface_temps_3)


    
def mean(L) :
    return sum(L) / len(L)

def var(L) :
    m = mean(L)
    return mean([(x - m) **2 for x in L])

surface_temp_residuals = surface_temps_3

print(f"Mean of surface temperature residuals : {mean(surface_temp_residuals)}")
print(f"Variance of surface temperature residuals : {var(surface_temp_residuals)}")


#plt.show()











