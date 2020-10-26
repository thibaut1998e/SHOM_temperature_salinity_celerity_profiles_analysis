import pickle
import matplotlib.pyplot as plt
from utils import LIGHT_PROF, get_profiles
from utils import transform_string_date_into_integer




if __name__ == '__main__':
    PROFS, _, _= get_profiles()
    lats = [prof.lat for prof in PROFS]
    lons = [prof.lon for prof in PROFS]

    surface_temps = [prof.temp[-1] for prof in PROFS]

    dates = [prof.date for prof in PROFS]

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
    plt.show()