import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import random as rd
import gsw
from scipy.interpolate import interp1d


# HelloWorld
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

    def get_sound_speed(self, domain=None):
        """return the list of sound speed for each depths using the salinity, temperature and pressure
        if domain is not None, it will resample the list of sound speed on tgis domain
        the lowest value of domain sould be higher than the lowest value in self.depth and the highest value of domain
        should be lower than the highest value in self.depth"""
        CT = gsw.CT_from_pt(self.salt, self.temp)
        p = np.array(self.depth)/10 - 10.1325
        sound_speed = gsw.density.sound_speed(CT, self.salt, p)
        if domain is not None:
            f = interp1d(self.depth, sound_speed)
            return f(domain)
        return sound_speed


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


def make_grid(lat_min=31, lat_max=45, lat_step=1, long_min=-4, long_max=35,
              long_step=1):
    longitudes = np.arange(long_min, long_max, long_step)
    latitudes = np.arange(lat_min, lat_max, lat_step)
    grid = cartesian_product(longitudes, latitudes)
    return grid


def day_number_to_period_of_year(day):
    while day > 365.25:
        day -= 365.25
    return day


def transform_string_date_into_integer(date):
    """IN : string in the form AAAA-MM-JJ 00:00:00 representing a date
    OUT : integer, day 1 is 2013-01-01"""
    date = str(date)
    year, month, day = date.split('-')
    day, _ = day.split(' ')
    year, month, day = int(year), int(month), int(day)
    lenght_of_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    x = 365 * (year - 2013) + sum(lenght_of_months[:month - 1]) + day
    return x


def cartesian_product(x, y):
    """In : x, y two 1d array (or list)
    OUT : 2D array cartesian product af x and y"""
    x = np.array(x)
    y = np.array(y)
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def gauss(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def distance(p1, p2, order=2):
    dist = 0
    for i in range(len(p1)):
        dist += abs(p1[i] - p2[i]) ** order
    return dist ** (1 / order)


def compute_profiles_statistics(profiles):
    lats = get_lats(profiles)
    longs = get_longs(profiles)
    dates = get_date(profiles)
    depths = np.array([prof.depth for prof in profiles])
    plt.hist([max(d) for d in depths])
    plt.show()
    min_long_lat = min(min(longs), min(lats))
    max_long_lat = max(max(longs), max(lats))
    max_depths = max([max(d) for d in depths])
    min_depths = min([max(d) for d in depths])
    print('min longitude', min(longs))
    print('max longitude', max(longs))
    print('min latitude', min(lats))
    print('max latitude', max(lats))
    print('max depths', max_depths)
    print('min depths', min_depths)
    return min_long_lat, max_long_lat, min_depths, max_depths, min(dates), max(dates)




def mean(L):
    return sum(L) / len(L)


def var(L):
    m = mean(L)
    return mean([(x - m) ** 2 for x in L])


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
        d = nb_points / (np.pi * radius ** 2)
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
                                      long_min=-4, long_max=35, long_step=0.1, lat_min=31, lat_step=0.1):
    """returns the density of profiles at any point on the map, by finding the nearest neighbour of the point
    it assumes that the grid of profile_densities is defined by long_min, long_max, long_step, lat_min, lat_max, lat_step.
    ex : point [5.87, 41.23] : it will return the density of profile at location [5.9, 41.2]"""

    long_nn, lat_nn = int(point[0] / long_step + 0.5) * long_step, int(point[1] / lat_step + 0.5) * lat_step
    idx = (lat_nn - lat_min) / lat_step * (long_max - long_min) / long_step + (long_nn - long_min) / long_step
    idx = int(idx)
    return profile_densities[idx]


def get_profiles(file='ts_profiles.pkl', p=0.01, density_grid=None, proba='sqrt'):
    """return the list of profiles in file and two set train_PROFS an valid_PROFS
     if density_grid is None it selects the profiles in valid_PROFS following an uniform probability distribution
     otherwise each profile is selected with a probability inversly proportional to the density of profiles at its location
     p is the proportion of validation data to use"""
    # p = proportion of validation data
    with open(file, 'rb') as f:
        PROFS = pickle.load(f)
    n = len(PROFS)
    if density_grid is None:
        np.random.shuffle(PROFS)
        valid_PROFS = PROFS[:int(n * p)]
        train_PROFS = PROFS[int(n * p):]

    else:
        nb_validation = n * p
        long_lats = get_long_lats(PROFS)
        densities = [density_profiles_nn_interpolation(density_grid, p) for p in long_lats]
        print(f"Mean density over all points : {mean(densities)}")
        if proba == 'proportional':
            C = nb_validation / sum(
                [1 / d for d in densities])  # we choose C such that the expectation of the number of validation
            # profiles is nb_validation = number of profiles * proportion of validation data
        elif proba == 'sqrt':
            C = nb_validation / sum([1 / np.sqrt(d) for d in densities])
        else:
            raise Exception('choose proba = proportional or sqrt')
        valid_PROFS = []
        valid_PROFS_idx = []
        for i in range(len(PROFS)):
            if proba == 'proportional':
                select_prob = C / densities[i]
            elif proba == 'sqrt':
                select_prob = C / np.sqrt(densities[i])
            else:
                raise Exception('choose proba = proportional or sqrt')
            if rd.random() < select_prob:
                valid_PROFS_idx.append(i)
                valid_PROFS.append(PROFS[i])
        print(f"Mean density over the validation set : {mean([densities[ind] for ind in valid_PROFS_idx])}")
        train_PROFS = [PROFS[idx] for idx in range(n) if idx not in valid_PROFS_idx]
    print('number of validation profiles', len(valid_PROFS))
    return PROFS, train_PROFS, valid_PROFS


def plot_density_histogram(PROFS, density_grid, title, range=None):
    long_lats = get_long_lats(PROFS)
    densities = [density_profiles_nn_interpolation(density_grid, p) for p in long_lats]
    plt.hist(densities, range=range)
    plt.xlabel('profile density')
    plt.ylabel('nb of profiles')
    plt.title(title)
    plt.show()






if __name__ == '__main__':
    PROFS, _, _ = get_profiles()
    compute_profiles_statistics(PROFS)
    from plot_profile_locations import plot_data

    #
    # density_profiles(PROFS, radius=1, lat_step=0.1, long_step=0.1) #save txt file with densities on a gridy
    long_lats, densities = read_density_txt()  # get the saved values of densities
    print('début')
    PROFS, _, valid_PROFS = get_profiles(p=0.05, density_grid=densities)
    plot_density_histogram(valid_PROFS, densities,
                           title='validation profile densities with probability selection inversly proportional to '
                                 'square root of density',
                           range=(0, 100))

    plot_data(get_longs(PROFS), get_lats(PROFS), [1 if p in valid_PROFS else 0 for p in PROFS],
              title='validation data location with probability of selction inversly proportional to squared root of density')
    PROFS, _, valid_PROFS = get_profiles(p=0.05)
    plot_density_histogram(valid_PROFS, densities,
                           title='validation profile densities with uniform probability of selection')
    plot_data(get_longs(PROFS), get_lats(PROFS), [1 if p in valid_PROFS else 0 for p in PROFS],
              title='validation data location with uniform probability of selction')

    long = 22.31
    lat = 34.65
    point = np.array([long, lat])
    density_at_point = density_profiles_nn_interpolation(densities, point)

    print(f'the density of profiles at point {point} is : {density_at_point}')

    print('plot profiles densities')

    plot_data(long_lats.T[0], long_lats.T[1], densities, color_log_scale=True, title='density of profiles (log scale)',
              save_location='density_of_profiles')

    longs = np.arange(-4, 35, 0.01)
    points = [[long, 38] for long in longs]
    plt.plot(longs,
             [density_profiles_nn_interpolation(densities, p) for p in points])  # plot profiles density at lat 38
    plt.show()














