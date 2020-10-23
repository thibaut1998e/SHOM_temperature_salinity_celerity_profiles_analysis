import numpy as np
import pickle
import  time
import matplotlib.pyplot as plt

class Point():
    def __init__(self, lon, lat, depth, date, temp=None, salt=None):
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.date = date #integer form of the date
        self.temp = temp
        self.salt = salt

    def __str__(self):
        return f' longitude {self.lon}, latitude {self.lat}, depth {self.depth}, date : {self.date}, ' \
              f'temperature {self.temp}, salinity {self.salt}'

    def distance_to(self, other):
        return distance(self, other)

    def gaussian_weights_interpolation(self, list_points, sigma):
        """compute the value of temp and salt using interpolation with gaussian weights on the points in list_points
        sigma : standard deviation of the gaussian
        """
        self.normalize()
        for p in list_points:
            p.normalize()

        weights = [gauss(self.distance_to(p), sigma) for p in list_points]

        S = sum(weights)

        weights = np.array([w/S for w in weights])
        temps = np.array([p.temp for p in list_points])
        salts = np.array([p.salt for p in list_points])
        self.temp = temps.dot(weights.T)
        self.salt = salts.dot(weights.T)

    def gaussian_weights_interpolation_with_profiles(self, profiles, sigma):
        list_points = []
        for prof in profiles:
            list_points = list_points + prof.get_list_of_points()
        self.gaussian_weights_interpolation(list_points, sigma)

    def normalize(self, min_long_lat=-4, max_long_lat=44, min_depth=0, max_depth=2500, min_date=0, max_date=365):
        self.lon = (self.lon-min_long_lat)/(max_long_lat-min_long_lat)
        self.lat = (self.lat-min_long_lat)/(max_long_lat-min_long_lat)
        self.depth = (self.depth - min_depth)/(max_depth-min_depth)
        self.date = (self.date - min_date)/(max_date - min_date)


def gauss(x, sigma):
    return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))


def distance(p1, p2, coef_lon=1, coef_lat=1, coef_depth=1, coef_date=1, power=2):
    dist = coef_lon*(p1.lon-p2.lon)**power + coef_lat*(p1.lat-p2.lat)**power + \
           coef_depth*(p1.depth-p2.depth)**power + coef_date*(p1.date-p2.date)**power
    return dist**(1/power)


def transform_string_date_into_integer(date):
    """IN : string in the form AAAA-MM-JJ 00:00:00 representing a date
    OUT : integer, day 1 is 2013-01-01"""
    year, month, day = date.split('-')
    day, _ = day.split(' ')
    year, month, day = int(year), int(month), int(day)
    lenght_of_months = [31,28,31,30,31,30,31,31,30,31,30,31]
    x = 365*(year-2013) + sum(lenght_of_months[:month-1]) + day
    return x


def compute_profiles_statistics(profiles):
    lats = np.array([prof.lat for prof in profiles])
    longs = np.array([prof.lon for prof in profiles])
    dates = np.array([transform_string_date_into_integer(str(prof.date)) for prof in profiles])
    depths = np.array([prof.depth for prof in profiles])
    min_long_lat = min(min(longs), min(lats))
    max_long_lat = max(max(longs), max(lats))
    max_depths = max([max(d) for d in depths])
    min_depths = min([min(d) for d in depths])
    print('min longitude', min(longs))
    print('max longitude', max(longs))
    print('min latitude', min(lats))
    print('max latitude', max(lats))
    print('max depths', max_depths)
    print('min depths', min_depths)

    return min_long_lat, max_long_lat, min_depths, max_depths, min(dates), max(dates)








if __name__ == '__main__':
    from plot_data import get_profiles, LIGHT_PROF
    PROFS = get_profiles()

    #plt.plot(range(len(PROFS)), [len(p.depth) for p in PROFS])
    #plt.show()
    compute_profiles_statistics(PROFS)
    p = Point(lon=10, lat=35, depth=1000, date=150)
    t = time.time()
    p.gaussian_weights_interpolation_with_profiles(PROFS[:1000], sigma=1)
    print('time to compute interpolation', time.time()-t)
    print(f'estimated temperature', p.temp)
    print('estimated salinity', p.salt)








