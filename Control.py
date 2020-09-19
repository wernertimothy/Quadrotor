import numpy as np
import scipy.linalg
import scipy.sparse
import cvxpy as cp
import osqp

import matplotlib.pylab as plt

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
            self._BoxConstraints[i] = np.array([-np.inf, np.inf])

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
                 the_A,         # discrete system matrix (should be scipy.sparse)            
                 the_B,         # discrete input matrix (should be scipy.sparse)
                 the_N,         # discrete horizon
                 the_Q,         # state penalty (should be scipy.sparse)
                 the_R,         # input penalty (should be scipy.sparse)
                 the_x_min,     # lower bound on x (should be np.array) use -np.inf/np.inf if unconstraint
                 the_x_max,     # upper bound on x
                 the_u_min,     # lower bounf on u
                 the_u_max      # upper bound on u
                 ):
        self.__N  = the_N                  
        self.__A  = the_A                  
        self.__B  = the_B                  
        self.__Q  = the_Q                  
        self.__R  = the_R 
        self.__xmin = the_x_min
        self.__xmax = the_x_max
        self.__umin = the_u_min
        self.__umax = the_u_max

        [self.__n, self.__p] = the_B.shape # state and input dimension

        

        # prealocate logging array
        self.predictedStateTrajectory = np.zeros((self.__n, self.__N+1))
        self.predictedInputTrajectory = np.zeros((self.__p, self.__N))

        # define optimization variable
        self.__U = cp.Variable((self.__p, self.__N))   # input trajectory
        self.__X = cp.Variable((self.__n, self.__N+1)) # state trajectory
        self.__IC = cp.Parameter(self.__n)             # set initial condition as parameter

        # build problem
        self.__buildProblem()
        
    def __buildProblem(self):
        objective = 0                                                 # initialize objective
        constraints =  [self.__X[:,0]        == self.__IC]            # first stage is current state
        constraints += [self.__X[:,self.__N] == np.zeros(self.__n)]   # last stage is zero (ZTC)
        # loop through stage 0 to N:
        for k in range(0, self.__N):
            # cost
            objective += cp.quad_form(self.__X[:,k], self.__Q) + cp.quad_form(self.__U[:,k], self.__R)
            # equality constraints
            constraints += [self.__X[:,k+1] == self.__A@self.__X[:,k] + self.__B@self.__U[:,k]]
            # inequality constraints
            constraints += [self.__xmin <= self.__X[:,k], self.__X[:,k] <= self.__xmax]
            constraints += [self.__umin <= self.__U[:,k], self.__U[:,k] <= self.__umax]
        # stage N+1:
        objective += cp.quad_form(self.__X[:,self.__N], self.__Q)

        self.__prob = cp.Problem(cp.Minimize(objective), constraints)

    def reshapeSolution(self):
        self.predictedStateTrajectory = self.__X.value
        self.predictedInputTrajectory = self.__U.value
        
    def run(self, the_state):
        self.__IC.value = the_state   # update problem with new initial state
        self.__prob.solve(verbose = True, warm_start = True, solver = cp.OSQP)
        self.reshapeSolution()
        return self.predictedInputTrajectory[:,0]
