import numpy as np

class PlanarQuadrotor:
    # === constructor ===
    def __init__(self):
        # === properties ===
        # == system properties ==
        self._StateDimension      = 6
        self._InputDimension      = 2
        self.x_num                = 1
        self.y_num                = 2
        self.theta_num            = 3
        self.dx_num               = 4
        self.dy_num               = 5
        self.dtheta_num           = 6
        # == parameter ==
        self.__mass               = 0.5    # [kg]     mass of the drone
        self.__inertia            = 0.002  # [kgm²]   moment of inertia about the center
        self.__ArmLength          = 0.1    # [m]      length from center to rotor
        self.__g                  = 9.81   # [m/s²]   gravity
        # == ststes ==
        # self._x                   = 0.0    # [m]      x position 
        # self._dx                  = 0.0    # [m/s]    x velocity
        # self._y                   = 0.0    # [m]      y position
        # self._dy                  = 0.0    # [m/s]    x velocity
        # self._theta               = 0.0    # [rad]    angle of drone
        # self._dtheta              = 0.0    # [rad/s]  angular velocity of drone
        # self._state               = { '1' : 0,
        #                               '2' : 0,
        #                               '3' : 0,
        #                               '4' : 0,
        #                               '5' : 0,
        #                               '6' : 0 }
        self._state               = np.zeros(self._StateDimension)
        # == integrator prperties ==
        self.SamleRate            = 0.01   # [s]      sample rate of the simualtion
        self.__IntegratorStepSize = 0.001  # [s]      step size of the integration
    # === methods ===
    def __evaluateRHS(self, the_state, the_input):
        stateDerivative = np.zeros_like(the_state)
        stateDerivative[self.x_num-1     ] = the_state[self.dx_num-1]
        stateDerivative[self.y_num-1     ] = the_state[self.dy_num-1]
        stateDerivative[self.theta_num-1 ] = the_state[self.dtheta_num-1]
        stateDerivative[self.dx_num-1    ] = -1/self.__mass*(the_input[0] + the_input[1])*np.sin(self._state[self.theta_num-1])
        stateDerivative[self.dy_num-1    ] =  1/self.__mass*(the_input[0] + the_input[1])*np.cos(self._state[self.theta_num-1]) - self.__g
        stateDerivative[self.dtheta_num-1] = self.__ArmLength/self.__inertia*(the_input[0]- the_input[1])
        return stateDerivative

    def set_state(self, the_state):
        self._state = the_state

    def set_SampleRate(self, the_rate):
        self.SamleRate = the_rate

    def Integrate(self, the_input, *argv):
        if not argv:
            TimeSpan = self.SamleRate
        else:
            TimeSpan = argv
        dt = self.__IntegratorStepSize
        t = 0
        while t < TimeSpan:
            k1 = self.__evaluateRHS(self._state,         the_input)
            k2 = self.__evaluateRHS(self._state+dt/2*k1, the_input)
            k3 = self.__evaluateRHS(self._state+dt/2*k2, the_input)
            k4 = self.__evaluateRHS(self._state+dt*k3,   the_input)
            self._state = self._state + dt/6*(k1+2*k2+2*k3+k4)
            t += dt

    def getLinearization(self):
            theta_bar  = 0.0
            u_bar = self.__mass*self.__g/2

            delddx_dtheta = -1/self.__mass*np.cos(theta_bar)*(u_bar + u_bar)
            delddy_dtheta =  1/self.__mass*np.sin(theta_bar)*(u_bar + u_bar)

            A = np.matrix([
                [0, 0,             0, 1, 0, 0],
                [0, 0,             0, 0, 1, 0],
                [0, 0,             0, 0, 0, 1],
                [0, 0, delddx_dtheta, 0, 0, 0],
                [0, 0, delddy_dtheta, 0, 0, 0],
                [0, 0,             0, 0, 0, 0]
            ])

            B = np.matrix([
                [0, 0],
                [0, 0],
                [0, 0],
                [-1/self.__mass*np.sin(theta_bar), -1/self.__mass*np.sin(theta_bar)],
                [ 1/self.__mass*np.cos(theta_bar),  1/self.__mass*np.cos(theta_bar)],
                [self.__ArmLength/self.__inertia , -self.__ArmLength/self.__inertia]
            ])

            return A, B