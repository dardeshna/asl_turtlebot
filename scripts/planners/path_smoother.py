import numpy as np
import scipy.interpolate
from scipy.interpolate import splrep, splev

def distance(x, y):
    return sum([(x[i] - y[i]) ** 2 for i in range(len(x))]) ** .5


def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    nominal_times = [0]
    t = 0
    path = np.asarray(path)
    for i in range(1, len(path)):
        time = distance(path[i,:], path[i - 1,:]) / V_des
        t += time
        nominal_times.append(t)
    t_smoothed = np.arange(start=0, stop=nominal_times[-1], step=dt)
    
    x_tck = splrep(nominal_times, path[:, 0], s=alpha)
    x_d = splev(t_smoothed, x_tck) 
    xd_d = splev(t_smoothed, x_tck, der=1) 
    xdd_d = splev(t_smoothed, x_tck, der=2) 

    y_tck = splrep(nominal_times, path[:, 1], s=alpha)
    y_d = splev(t_smoothed, y_tck)  
    yd_d = splev(t_smoothed, y_tck, der=1)  
    ydd_d = splev(t_smoothed, y_tck, der=2)  

    theta_d = np.arctan2(yd_d, xd_d)
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed
