from Systems import PlanarQuadrotor
from Control import ZTC_MPC

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# define system
quad = PlanarQuadrotor()
X0 = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0 ]) # initial condition
quad.set_state(X0)                              # set initial condition
quad.set_SampleRate(0.01)                       # set the sample rate

# define controler
A, B = quad.getDiscreteLinearization()
Q = np.diag([10, 10, 0.1, 1, 1, 1])
R = np.diag([10, 10])
N = 50
ctrl = ZTC_MPC(A, B, N, Q, R)
u = ctrl.run(X0)

plt.plot(ctrl.predictedStateTrajectory[0,:], ctrl.predictedStateTrajectory[1,:])
plt.show()
