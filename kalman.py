from utils import *


H = np.array([[1,0,0]])
std_mes = 1
std_dyn = 1


def Immersion(celerity_prof, k):
    return abs(celerity_prof.domain[k] - celerity_prof.domain[k-1])


def phi(celerity_prof, k):
    Im = Immersion(celerity_prof, k)
    transition_matrix = np.array([[1, Im, Im**2/2],
                                  [0, 1, Im],
                                  [0, 0, 1]])
    return transition_matrix


def Q(celerity_prof, k):
    Im = Immersion(celerity_prof, k)
    cov_mat = std_dyn ** 2 * np.array([[Im ** 6 / 20, Im ** 5 / 8, Im ** 4 / 6],
                                       [Im ** 5 / 8, Im ** 4 / 3, Im ** 3 / 2],
                                       [Im ** 4 / 6, Im ** 3 / 2, Im ** 2]])
    return cov_mat


def kalman_filter(celerity_prof):
    celerities = celerity_prof.celerities
    k_start = len(celerities)-1
    X_kk = np.array([celerities[-3:]]).T
    #print('shape', X_kk.shape)
    P_kk = Q(celerity_prof, k_start)
    filtered_signal = [X_kk[0]]
    for k in range(k_start, 0, -1):
        transition_mat = phi(celerity_prof, k)
        X_k1k = transition_mat.dot(X_kk)
        P_k1k = transition_mat.dot(P_kk).dot(transition_mat.T) + Q(celerity_prof, k)
        M = H.dot(P_k1k).dot(H.T) + std_mes
        K = P_k1k.dot(H.T).dot(np.linalg.inv(M))
        z = celerities[k-1]
        X_kk = X_k1k + K.dot(z-H.dot(X_k1k))
        P_kk = (np.eye(3) - K.dot(H)).dot(P_k1k)
        filtered_signal.append(X_kk[0])
    celerity_prof.celerities = filtered_signal
    return filtered_signal


PROFS = get_all_profiles()
prof1 = PROFS[0]
prof1.clean()

celerity_prof = prof1.get_sound_speed()
celerity_prof.plot()
filtered_signal = kalman_filter(celerity_prof)
celerity_prof.plot()



