import pickle
import numpy as np
import os
import  matplotlib.pyplot as plt


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


def get_long_lats(PROFS):
    """returns a 2D array of shape (2, N_profiles), set of (long, lat) coordinates"""
    return np.array([get_longs(PROFS), get_lats(PROFS)]).T


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


def get_profiles(p=0.05):
    #p = proportion of validation data
    with open('ts_profiles.pkl', 'rb') as f:
        PROFS = pickle.load(f)
    np.random.shuffle(PROFS)
    n = len(PROFS)
    valid_PROFS = PROFS[:int(n*p)]
    train_PROFS = PROFS[int(n*p):]
    return PROFS, train_PROFS, valid_PROFS


def make_grid(lat_min=32, lat_max=44, lat_step=1, long_min=-4, long_max=35,
                      long_step=1):
    longitudes = np.arange(long_min, long_max, long_step)
    latitudes = np.arange(lat_min, lat_max, lat_step)
    grid = cartesian_product(longitudes, latitudes)
    return grid


def density_profiles(PROFS, radius=1, save_loc='profile_density.txt', **args_grid):
    """save the densities of profiles in the file save location
    first a grid is created (with thee arguments **args_grid), then for each point of the grid
    the density is computed by counting the number profiles located in a circle (radius odf the circle given in parameter)"""
    if os.path.exists(save_loc):
        q = input(f'delete file {save_loc} (y)')
        if q == 'y':
            os.remove(save_loc)
        else:
            exit()
    f = open(save_loc, "w")
    print('long\tlat\tdensity', file=f)
    long_lats = get_long_lats(PROFS)
    grid = make_grid(**args_grid)
    print('number of point', len(grid))
    cpt = 0
    for point in grid:
        cpt += 1
        print(cpt)
        nb_points = 0
        for p in long_lats:
            if distance(point, p) < radius:
                nb_points += 1
        d = nb_points / (np.pi*radius**2)
        long, lat = point
        print(f'{long}\t{lat}\t{d}', file=f)


def read_density_txt(txt_path='profile_density.txt'):
    """read the txt file saved by the function density profiles, return a 2D array shape (2, Nb_profiles) containing lat/long
    coordinates of the grid, and a 1D array containing the associated density"""
    with open(txt_path) as f:
        lines = f.readlines()
    longs_lats = []
    densities = []
    for line in lines[1:]:
        long, lat, d = line.split('\t')
        long, lat, d = float(long), float(lat), float(d[:-2])
        longs_lats.append([long, lat])
        densities.append(d)
    return np.array(longs_lats), np.array(densities)


def density_profiles_nn_interpolation(profile_densities, point,
                                      long_min=-4, long_max=35, long_step=0.1, lat_min=32, lat_step=0.1):
    """returns the density of profiles at any point on the map, by finding the nearest neighbour of the point
    it assumes that the grid of profile_densities is defined by long_min, long_max, long_step, lat_min, lat_max, lat_step.
    ex : point [5.87, 41.23] : it will return the density of profile at location [5.9, 41.2]"""
    long_nn, lat_nn = int(point[0]/long_step+0.5)*long_step, int(point[1]/lat_step+0.5)*lat_step
    idx = (lat_nn - lat_min)/lat_step * (long_max-long_min)/long_step + (long_nn - long_min)/long_step
    idx = int(idx)
    return profile_densities[idx]




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
    """In : x, y two 1d array (or list)
    OUT : 2D array cartesian product af x and y"""
    x = np.array(x)
    y = np.array(y)
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




if __name__ == '__main__':

    PROFS, _, _ = get_profiles()
    #density_profiles(PROFS, radius=1, lat_step=0.1, long_step=0.1) #save txt file with densities on a grid
    long_lats, densities = read_density_txt() #get the saved values of densities
    long = 22.31
    lat = 34.65
    point = np.array([long, lat])
    density_at_point = density_profiles_nn_interpolation(densities, point)

    print(f'the density of profiles at point {point} is : {density_at_point}')
    from plot_profile_locations import plot_data
    print('plot profiles densities')
    plot_data(long_lats.T[0], long_lats.T[1], densities, color_log_scale=True, title='density of profiles (log scale)',
              save_location='density_of_profiles')










