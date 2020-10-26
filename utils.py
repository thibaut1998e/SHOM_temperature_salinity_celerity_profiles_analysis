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


def get_profiles(p=0.005):
    #p = proportion of validation data
    with open('ts_profiles.pkl', 'rb') as f:
        PROFS = pickle.load(f)
    np.random.shuffle(PROFS)
    n = len(PROFS)
    valid_PROFS = PROFS[:int(n*p)]
    train_PROFS = PROFS[int(n*p):]
    return PROFS, train_PROFS, valid_PROFS



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


def cartesian_product(x, y):
    """In : x, y two 1d array
    OUT : 2D array cartesian product af x and y"""
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def gauss(x, sigma):
    return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))


def distance(p1, p2, order=2):
    dist = 0
    for i in range(len(p1)):
        dist += abs(p1[i]-p2[i])**order
    return dist**(1/order)


def compute_profiles_statistics(profiles):
    lats = get_lats(profiles)
    longs = get_longs(profiles)
    dates = get_date(profiles)
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

def mean(L) :
    return sum(L) / len(L)

def var(L) :
    m = mean(L)
    return mean([(x - m) **2 for x in L])





