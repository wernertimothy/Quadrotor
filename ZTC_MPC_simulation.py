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


# simulate
simulation_time = 0.01
sim_N = int(simulation_time/quad.SamleRate)

X = np.empty([quad._StateDimension,sim_N])
U = np.empty([quad._InputDimension,sim_N])

time = np.arange(0.0, simulation_time, quad.SamleRate)
step = 0

X[:,step] = X0
for step,t in enumerate(time):
    u = ctrl.run(X[:,step]) + np.array([0.25*9.81, 0.25*9.81])  # run controler
    U[:,step] = u                        # log input
    quad.Integrate(u)                    # apply input to system
    X[:,step] = quad._state              # log the state


plt.figure(1)
# plt.plot(ctrl.predictedStateTrajectory[0,:], ctrl.predictedStateTrajectory[1,:])
plt.plot(X[0,:], X[1,:])

fig2, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(time, U[0,:])
ax2.plot(time, U[1,:])
ax1.set_ylabel('u1 [N]')
ax2.set_ylabel('u2 [N]')
ax2.set_xlabel('time [s]')
ax1.grid()
ax2.grid()

plt.show()
