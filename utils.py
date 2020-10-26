import pickle
import numpy as np

#HelloWorld
# profiles class
class LIGHT_PROF():
    def __init__(self, temp, salt, depth, lon, lat, date):
        self.salt = salt
        self.temp = temp
        self.depth = depth
        self.lon = lon
        self.lat = lat
        self.date = date

    def surface_temp(self):
        return self.temp[-1]


def get_longs(PROFS):
    return np.array([p.lon for p in PROFS])


def get_lats(PROFS):
    return np.array([p.lat for p in PROFS])


def get_surf_salt(PROFS):
    return np.array([p.salt[-1] for p in PROFS])


def get_surf_temp(PROFS):
    return np.array([p.temp[-1] for p in PROFS])


def get_date(PROFS):
    return np.array([transform_string_date_into_integer(p.date) for p in PROFS])


def get_profiles():
    with open('ts_profiles.pkl', 'rb') as f:
        PROFS = pickle.load(f)
    return PROFS


def day_number_to_period_of_year(day) :
    while day > 365.25 :
        day -= 365.25
    return day


def transform_string_date_into_integer(date):
    """IN : string in the form AAAA-MM-JJ 00:00:00 representing a date
    OUT : integer, day 1 is 2013-01-01"""
    date = str(date)
    year, month, day = date.split('-')
    day, _ = day.split(' ')
    year, month, day = int(year), int(month), int(day)
    lenght_of_months = [31,28,31,30,31,30,31,31,30,31,30,31]
    x = 365*(year-2013) + sum(lenght_of_months[:month-1]) + day
    return x





