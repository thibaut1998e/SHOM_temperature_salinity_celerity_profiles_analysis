import pickle
import matplotlib.pyplot as plt


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
        return


def get_profiles():
    with open('ts_profiles.pkl', 'rb') as f:
        PROFS = pickle.load(f)
    return PROFS


# load profiles
if __name__ == '__main__':


    # one profile plot
    i = 5000
    prof = PROFS[i]
    plt.plot(prof.temp, -prof.depth)

