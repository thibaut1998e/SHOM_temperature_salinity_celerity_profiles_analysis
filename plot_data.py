import pickle
import matplotlib.pyplot as plt
from Interpolation import transform_string_date_into_integer, Point


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

    def get_list_of_points(self):
        """return associated list of points"""
        date = transform_string_date_into_integer(str(self.date))
        points = [Point(self.lon, self.lat, self.depth[i], date, self.temp[i], self.salt[i])
                  for i in range(len(self.depth))]
        return points


def get_profiles():
    with open('ts_profiles.pkl', 'rb') as f:
        PROFS = pickle.load(f)
    return PROFS


# load profiles
if __name__ == '__main__':
    PROFS = get_profiles()
    i = 5000
    prof = PROFS[i]
    plt.plot(prof.temp, -prof.depth)

