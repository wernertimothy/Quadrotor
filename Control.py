import numpy as np
import scipy.linalg
import cvxpy as cp
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

        self.__n   = np.size(self.__A,0)                         # state dimension
        self.__p   = np.size(self.__B,1)                         # input dimension
        self.__dim = (self.__N+1)*self.__n + self.__N*self.__p   # QP dimension

        self.__IC = np.zeros(self.__n)

        self.predictedStateTrajectory = np.zeros((self.__n, self.__N+1))
        self.predictedInputTrajectory = np.zeros((self.__p, self.__N))

        self.__H = np.zeros((self.__dim, self.__dim))
        self.__Aeq = np.zeros((self.__n*(self.__N+1),self.__dim))
        self.__beq = np.zeros(self.__n*(self.__N+1))

        self.__Z = cp.Variable((self.__N+1)*self.__n + self.__N*self.__p)

        self.__buildCost()
        self.__buildEqualityConstraints()
        # self.__buildInequalityConstraints()

    def __buildCost(self):
        for k in range(0,self.__N):
            # stack Q on the diagonal from stage 0 to N-1
            self.__H[k*self.__n:(k+1)*self.__n,
                     k*self.__n:(k+1)*self.__n] = 2*self.__Q
            # stack R on the diagonal from stage 0 to N-1
            self.__H[self.__n*(self.__N+1)+k*self.__p:self.__n*(self.__N+1)+(k+1)*self.__p,
                     self.__n*(self.__N+1)+k*self.__p:self.__n*(self.__N+1)+(k+1)*self.__p] = 2*self.__R
        # stack Q for stage N
        self.__H[self.__N*self.__n:(self.__N+1)*self.__n,
                 self.__N*self.__n:(self.__N+1)*self.__n] = 2*self.__Q

    def __buildEqualityConstraints(self):
        # x(j) = intial condition
        self.__Aeq[0:self.__n, 0:self.__n] = np.identity(self.__n)
        self.__beq[0:self.__n] = self.__IC
        # Ax(j+k) - Ix(j+k+1) + Bu(j+k) = 0
        for k in range(0, self.__N):
            self.__Aeq[(k+1)*self.__n:(k+2)*self.__n,
                       k*self.__n:(k+2)*self.__n] = np.concatenate((self.__A, -np.identity(self.__n)), axis = 1)
            self.__Aeq[(k+1)*self.__n:(k+2)*self.__n,
                       self.__n*(self.__N+1)+k*self.__p:self.__n*(self.__N+1)+(k+1)*self.__p] = self.__B

    def __buildInequalityConstraints(self):
        pass

    def __buildProblem(self):
        self.__prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(self.__Z, self.__H)),
                                 self.__Aeq @ self.__Z == self.__beq)

    def setInitialCondition(self, the_initialCondition):
        self.__IC = the_initialCondition

    def updateProblem(self):
        self.__beq[0:self.__n] = self.__IC # set new initial condition
        self.__buildProblem()              # rebuild the problem

    def reshapeSolution(self):
        pass
        
    def run(self, the_state):
        self.setInitialCondition(the_state)
        self.updateProblem()
        self.__prob.solve()
        self.reshapeSolution()
        return self.predictedInputTrajectory[:,0]
