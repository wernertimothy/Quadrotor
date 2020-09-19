from Systems import PlanarQuadrotor
from Control import ZTC_MPC

import numpy as np
import cvxpy as cp
import scipy.sparse
import matplotlib.pyplot as plt

# define system
quad = PlanarQuadrotor()
X0 = np.array([float(-1.0 + np.random.rand(1)*2),
               float(-1.0 + np.random.rand(1)*2),
               float(-1.5 + np.random.rand(1)*3),
               float(-0.5 + np.random.rand(1)*1),
               float(-0.5 + np.random.rand(1)*1),
               float(-0.1 + np.random.rand(1)*0.2) ])    # random initial condition
# X0 = np.array([0.3, -0.5, 0.0, 0.0, 0.0, 0.0 ])          # initial condition
quad.set_state(X0)                                       # set initial condition
quad.set_SampleRate(0.01)                                # set the sample rate

# define controler
A, B = quad.getDiscreteLinearization()
A = scipy.sparse.csr_matrix(A)
B = scipy.sparse.csr_matrix(B)
Q = scipy.sparse.diags([10, 10, 0.1, 1, 1, 1])
R = scipy.sparse.diags([10, 10])
N = 200
xmin = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
xmax = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
umin = np.array([-np.inf, -np.inf])
umax = np.array([ np.inf,  np.inf])

ctrl = ZTC_MPC(A, B, N, Q, R, xmin, xmax, umin, umax)
u = ctrl.run(X0)
print(u)

plt.plot(ctrl.predictedStateTrajectory[0,:], ctrl.predictedStateTrajectory[1,:])
plt.show()
