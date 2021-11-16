import numpy as np
from utils import wrapToPi

from std_msgs.msg       import Float64

import rospy

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

        self.alpha = rospy.Publisher('/controller/alpha', Float64, queue_size = 10)
        self.delta = rospy.Publisher('/controller/delta', Float64, queue_size=10)
        self.rho = rospy.Publisher('/controller/rho', Float64, queue_size=10)

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########

        rho = np.sqrt((x-self.x_g)**2+(y-self.y_g)**2)
        delta = wrapToPi(np.arctan2(self.y_g-y, self.x_g-x)-self.th_g)
        alpha = wrapToPi(delta-(th-self.th_g))

        self.alpha.publish(alpha)
        self.rho.publish(rho)
        self.delta.publish(delta)

        V = self.k1*rho*np.cos(alpha)
        om = self.k2*alpha + self.k1*np.sinc(alpha/np.pi)*np.cos(alpha)*(alpha+self.k3*delta)
        
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
