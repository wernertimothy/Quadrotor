import numpy as np
import scipy.linalg
import scipy.sparse
import cvxpy as cp
import casadi

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

class ZTC_LMPC:
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
        self.__N     = the_N                  
        self.__A     = the_A                  
        self.__B     = the_B                  
        self.__Q     = the_Q                  
        self.__R     = the_R 
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
        self.__prob.solve(verbose = False, warm_start = True, solver = cp.OSQP )
        self.reshapeSolution()
        return self.predictedInputTrajectory[:,0]


class QINF_LMPC:
    def __init__(self,
                 the_A,         # discrete system matrix (should be scipy.sparse)            
                 the_B,         # discrete input matrix (should be scipy.sparse)
                 the_N,         # discrete horizon
                 the_Q,         # state penalty (should be scipy.sparse)
                 the_R,         # input penalty (should be scipy.sparse)
                 the_P,         # defines terminal region
                 the_alpha,     # defines terminal region
                 the_x_min,     # lower bound on x (should be np.array) use -np.inf/np.inf if unconstraint
                 the_x_max,     # upper bound on x
                 the_u_min,     # lower bounf on u
                 the_u_max      # upper bound on u
                 ):
        self.__N     = the_N                  
        self.__A     = the_A                  
        self.__B     = the_B                  
        self.__Q     = the_Q                  
        self.__R     = the_R
        self.__P     = the_P
        self.__alpha = the_alpha
        self.__xmin  = the_x_min
        self.__xmax  = the_x_max
        self.__umin  = the_u_min
        self.__umax  = the_u_max

        [self.__n, self.__p] = the_B.shape # state and input dimension

        # prealocate logging array
        self.predictedStateTrajectory = np.zeros((self.__n, self.__N+1))
        self.predictedInputTrajectory = np.zeros((self.__p, self.__N))

        # define optimization variables
        self.__opti = casadi.Opti()
        self.__U = self.__opti.variable(self.__p, self.__N)
        self.__X = self.__opti.variable(self.__n, self.__N+1)
        self.__IC = self.__opti.parameter(self.__n)

        # build problem
        self.__buildProblem()

    def __buildProblem(self):
        # define some functions:
        As = casadi.MX(self.__A)
        Bs = casadi.MX(self.__B)
        # xmins = casadi.MX(self.__xmin)
        # xmaxs = casadi.MX(self.__xmax)
        # umins = casadi.MX(self.__umin)
        # umaxs = casadi.MX(self.__umax)

        # initial condition
        self.__opti.subject_to(self.__X[:,0] == self.__IC)
        # terminal constraint
        self.__opti.subject_to(self.__X[:,self.__N].T@self.__P@self.__X[:,self.__N] <= self.__alpha)
        
        objective = 0 # init objective
        for k in range(0,self.__N):
            # stage cost:
            objective += self.__X[:,k].T@self.__Q@self.__X[:,k] 
            objective += self.__U[:,k].T@self.__R@self.__U[:,k]
            # dynamic constraints:
            self.__opti.subject_to(self.__X[:,k+1] == As@self.__X[:,k] + Bs@self.__U[:,k])
            # state constraints
            # self.__opti.subject_to(self.__xmin <= self.__X[:,k] <= self.__xmax)
            # input constraints
            # self.__opti.subject_to(self.__umin <= self.__U[:,k] <= self.__umax)
        # terminal cost
        objective += self.__X[:,self.__N].T@self.__P@self.__X[:,self.__N]

        self.__opti.minimize(objective)

        self.__opti.solver('ipopt')
        
    def reshapeSolution(self):
        self.predictedStateTrajectory = self.sol.value(self.__X)
        self.predictedInputTrajectory = self.sol.value(self.__U)

    def visualizeTerminalRegion(self):
        P = self.__P[0:2,0:2]
        T = scipy.linalg.sqrtm(scipy.linalg.inv(P)/self.__alpha)
        m = 100
        theta = np.linspace(0, 2*np.pi, m)
        x = np.cos(theta)
        y = np.sin(theta)
        # define circle
        C = np.zeros((2,m))
        C[0,:] = x
        C[1,:] = y
        # define ellipse
        E = np.zeros((2,m))
        for k in range(0,m):
            E[:,k] = T@C[:,k]
        return E
        
    def run(self, the_state):
        self.__opti.set_value(self.__IC, the_state)  # update problem with new initial state
        self.sol = self.__opti.solve()
        self.reshapeSolution()
        return self.predictedInputTrajectory[:,0]

def ComputeTerminalRegion(the_A, the_B, the_Q, the_R, the_umin, the_umax):
    # step 1: determine K using LQR
    P = np.matrix(scipy.linalg.solve_continuous_are(the_A, the_B, the_Q, the_R))
    K = np.array(scipy.linalg.inv(the_R)*(the_B.T*P)) 
    eigvals, eigvecs = np.linalg.eig(the_A - the_B@K)
    max_eig = np.max(eigvals)
    # choose now kappa sucht that lam_max(A-BK) < - kappa
    kappa = 0.9

    # # step 2: compute P from Lyapunov equation
    [n, p] = the_B.shape
    P = scipy.linalg.solve_continuous_lyapunov( the_A-the_B@K + kappa*np.identity(n), -(the_Q+K.T@the_R@K))

    # # step 3: min. x'Px
    # #         s.t. u_min <= u <= u_max

    # opti = casadi.Opti()
    # x = opti.variable(n)
    # objective = -x.T@P@x
    # opti.minimize(objective)
    # opti.subject_to(the_umin <= K@x <= the_umax)


    # P = scipy.sparse.csr_matrix(P)

    # Aiq = np.concatenate((np.identity(p), np.identity(p)))
    # # Aiq = scipy.sparse.csr_matrix(Aiq@K)

    # K = scipy.sparse.csr_matrix(K)

    # biq = np.concatenate((the_umin, the_umax))

    # m   = np.size(Aiq, 0)
    # Alpha = np.zeros(m)
    # q = np.zeros(n)

    # prob = osqp.OSQP()

    # for k in range(0,m):
    #     A = scipy.sparse.csr_matrix(Aiq@K)
    #     b = np.array([biq[k]])
    #     prob.setup(-P, q, A, biq, biq, alpha=1.0)
    #     Alpha[k] = prob.solve()

    alpha = 0.9

    return P, alpha     


class OutputTracking_LMPC():
    def __init__(
        self,
        the_A,          # discrete system matrix
        the_B,          # discrete input matrix
        the_C,          # output matrix
        the_Q,          # state penalty
        the_R,          # input penalty
        the_DR,         # input rate penalty
        the_P,          # terminal cost
        the_N,          # discrete horizon
        the_xmin,       # lower bounds on states
        the_xmax,       # upper bounds on states
        the_umin,       # lower bounds on inputs
        the_umax,       # upper bounds on input
        the_Dumin,      # lower bound in input rate
        the_Dumax,      # upper bound on input rate
        ):
        self.__A     = the_A                  
        self.__B     = the_B
        self.__C     = the_C                 
        self.__Q     = the_Q                  
        self.__R     = the_R
        self.__DR    = the_DR
        self.__P     = the_P
        self.__N     = the_N
        self.__xmin  = the_xmin
        self.__xmax  = the_xmax
        self.__umin  = the_umin
        self.__umax  = the_umax
        self.__Dumin = the_Dumin
        self.__Dumax = the_Dumax

        [self.__n, self.__p] = the_B.shape # state and input dimension
        self.__q = np.size(self.__C,0)     # output dimension

        # prealocate logging array
        self.predictedStateTrajectory     = np.zeros((self.__n, self.__N+1))
        self.predictedInputTrajectory     = np.zeros((self.__p, self.__N+1))
        self.predictedInputRateTrajectory = np.zeros((self.__p, self.__N))

        # define optimization variables
        self.__DU        = cp.Variable((self.__p, self.__N))    # input rate trajectory
        self.__U         = cp.Variable((self.__p, self.__N+1))  # input trajectory
        self.__X         = cp.Variable((self.__n, self.__N+1))  # state trajectory
        self.__InitState = cp.Parameter(self.__n)               # set initial state as parameter
        self.__InitInput = cp.Parameter(self.__p)               # set inital input
        self.__r         = cp.Parameter((self.__q, self.__N+1)) # set reference as parameter

        # build problem
        self.__buildProblem()

    def __buildProblem(self):
        objective   = 0                                        # initialize objective
        constraints =  [self.__X[:,0] == self.__InitState]     # first stage is current state
        constraints =  [self.__U[:,0] == self.__InitInput]     # first stage is last input
        # loop through stage 0 to N:
        for k in range(0, self.__N):
            # quadratic cost
            objective   += cp.quad_form(self.__X[:,k], self.__C.T@self.__Q@self.__C)
            objective   += cp.quad_form(self.__U[:,k], self.__R)
            objective   += cp.quad_form(self.__DU[:,k], self.__DR)
            # linear cost
            objective   += -2*self.__r[:,k].T@self.__Q@self.__C@self.__X[:,k]
            # equality constraints
            constraints += [self.__X[:,k+1] == self.__A@self.__X[:,k] + self.__B@self.__U[:,k]]
            constraints += [self.__DU[:,k]  == self.__U[:,k+1] - self.__U[:,k]]
            # inequality constraints
            constraints += [self.__xmin  <= self.__X[:,k],  self.__X[:,k]  <= self.__xmax]
            constraints += [self.__umin  <= self.__U[:,k],  self.__U[:,k]  <= self.__umax]
            constraints += [self.__Dumin <= self.__DU[:,k], self.__DU[:,k] <= self.__Dumax]
        # stage N+1:
        objective   += cp.quad_form(self.__X[:,self.__N], self.__C.T@self.__P@self.__C)
        objective   += cp.quad_form(self.__U[:,self.__N], self.__R)
        objective   += -2*self.__r[:,self.__N].T@self.__Q@self.__C@self.__X[:,self.__N]
        constraints += [self.__xmin  <= self.__X[:,self.__N],  self.__X[:,self.__N]  <= self.__xmax]
        constraints += [self.__umin  <= self.__U[:,self.__N],  self.__U[:,self.__N]  <= self.__umax]

        self.__prob = cp.Problem(cp.Minimize(objective), constraints)

    def reshapeSolution(self):
        self.predictedStateTrajectory     = self.__X.value
        self.predictedInputTrajectory     = self.__U.value
        self.predictedInputRateTrajectory = self.__DU.value
        
    def run(self, the_state, the_input, the_reference):
        self.__InitState.value = the_state      # update problem with new initial state
        self.__InitInput.value = the_input      # and last input
        self.__r.value         = the_reference  # and new reference

        self.__prob.solve(verbose = False, warm_start = True, solver = cp.OSQP )

        self.reshapeSolution()

        return self.predictedInputTrajectory[:,1]    
      