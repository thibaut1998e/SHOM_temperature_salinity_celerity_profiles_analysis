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

dates = [prof.date for prof in PROFS]



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

integer_dates = [transform_string_date_into_integer(str(date)) for date in dates]

# Plots all the surface temperatures recorded on the time axis
plt.scatter(integer_dates, surface_temps)

n = len(PROFS)
i = 1
mean_surface_temps = {}
sum_of_surface_temps = surface_temps[0]
number_of_profiles = 1
while i < n :
    if integer_dates[i] == integer_dates[i-1] :
        sum_of_surface_temps += surface_temps[i]
        number_of_profiles += 1
    else :
        mean_surface_temps[integer_dates[i-1]] = sum_of_surface_temps / number_of_profiles
        sum_of_surface_temps = surface_temps[i]
        number_of_profiles = 1
    i += 1

X = list(mean_surface_temps)
Y = [mean_surface_temps[x] for x in X]

# Plots the means of the recorded surface temperatures over time
plt.plot(X, Y)
















