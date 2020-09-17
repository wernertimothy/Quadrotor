import numpy as np
import scipy.linalg
from math import inf

class ContinuousLQR:
    def __init__(self,
                 A,   # system matrix \inR^{nxn}
                 B,   # input  matrix \inR^{nxp}
                 Q,   # state penalty Q = Q.T >= 0 \inR^{nxn}
                 R ): # input penylty R = R.T >  0 \inR^{pxp}
        self.__A = A
        self.__B = B
        self.__Q = Q
        self.__R = R
        self.__StateDimension = np.size(A,0)
        self.__InputDimension = np.size(B,1)
        self.K = np.zeros([self.__InputDimension, self.__StateDimension])
        self.ControlOffset = np.zeros(self.__InputDimension)

        self._BoxConstraints = {}
        for i in range(0, self.__InputDimension):
            self._BoxConstraints[i] = np.array([-inf, inf])

        if self.__check_conditions():
            P = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R)) # solve continous Riccati
            self.K = np.array(scipy.linalg.inv(R)*(B.T*P))               # calculate LQR gain

    def __check_conditions(self):
        # 1. for Q = C.TC, (A,C) should be observable:
        C = scipy.linalg.sqrtm(self.__Q)
        for n in range(0, self.__StateDimension):
            if n == 0:
                obsv = C
            else:
                obsv = np.concatenate( (obsv, np.linalg.matrix_power(self.__A,n)), axis = 0)
        if not np.linalg.matrix_rank(obsv) == self.__StateDimension:
            raise Exception('the pair (A,C) where C.TC = Q is not observable!')
        # 2. (A,B) should be stabilizable:
        eigvals, eigvecs = np.linalg.eig(self.__A)
        for eig in eigvals:
            if eig >= 0:
                Hautus = np.concatenate( (eig*np.identity(self.__StateDimension)-self.__A, self.__B), axis=1 )
                if not np.linalg.matrix_rank(Hautus) == self.__StateDimension:
                    raise Exception('at least one unstable eigenvalue of A is not stabilizable!')
        return True

    def setBoxConstraints(self, the_box):
        for i in range(0, self.__InputDimension):
            self._BoxConstraints[i] = the_box[i]
        
    def setControlOffset(self, the_offset):
        self.ControlOffset = the_offset

    def __saturation(self, the_u):
        for input, constraint in self._BoxConstraints.items():
            the_u[input] = constraint[0] if the_u[input] <= constraint[0] else the_u[input]
            the_u[input] = constraint[1] if the_u[input] >= constraint[1] else the_u[input]
        return the_u

    def runStabilizing(self, the_state):
        return self.__saturation(-self.K@the_state + self.ControlOffset)
        
    def runTracking(self, the_state, the_reference):
        return self.__saturation(self.K@(the_reference - the_state) + self.ControlOffset)

class ZTC_MPC:
    def __init__(self,
                 the_A,
                 the_B,
                 the_initialCondition,
                 the_N, 
                 the_Q, 
                 the_R, 
                 the_stateConstraints, 
                 the_inputConstraints ):
        self.__N  = the_N                 # discrete horizon
        self.__A  = the_A                 # discrete system matrix
        self.__B  = the_B                 # discrete input matrix
        self.__Q  = the_Q                 # state penalty
        self.__R  = the_R                 # input penalty
        self.__IC = the_initialCondition  # initial condition

        self.__n = np.size(self.__A,0)
        self.__p = np.size(self.__B,1)

        self.predictedStateTrajectory = np.zeros((self.__n, self.__N+1))
        self.predictedInputTrajectory = np.zeros((self.__p, self.__N))

        # self.__Z = cp.Variable((self.__N+1)*self.__n + self.__N*self.__p)

        # build up cost
        # buil up equality constraints
        # build up inequality constraints 

        self.__prob = 0

    def setInitialCondition(self, the_initialCondition):
        self.__IC = the_initialCondition

    def updateProblem(self):
        pass

    def reshapeSolution(self):
        pass
        
    def run(self, the_state):
        self.setInitialCondition(the_state)
        self.updateProblem()
        # self.__prob.solve()
        self.reshapeSolution()
        return self.predictedInputTrajectory[:,0]
