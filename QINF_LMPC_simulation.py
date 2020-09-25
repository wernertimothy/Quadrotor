from Systems import PlanarQuadrotor
from Control import QINF_LMPC
from Control import ComputeTerminalRegion

import numpy as np
import cvxpy as cp
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define system
quad = PlanarQuadrotor()
X0 = np.array([float(-1.0 + np.random.rand(1)*2),
               float(-1.0 + np.random.rand(1)*2),
               float(-np.pi/4 + np.random.rand(1)*np.pi/2),
               float(-0.5 + np.random.rand(1)*1),
               float(-0.5 + np.random.rand(1)*1),
               float(-0.2 + np.random.rand(1)*0.4) ])    # random initial condition
X0 = np.array([-0.3, -0.8, 0.6, 0.1, -0.8, 0.2 ])          # initial condition
quad.set_state(X0)                                       # set initial condition
quad.set_SampleRate(0.01)                                # set the sample rate

# define controler
xmin = np.array([-np.inf, -np.inf, -np.inf, -1.0, -1.0, -np.inf])
xmax = np.array([ np.inf,  np.inf,  np.inf,  1.0,  1.0,  np.inf])
umin = np.array([ 0.5, 0.5]) - np.array([0.25*9.81, 0.25*9.81])
umax = np.array([ 5.0, 5.0]) - np.array([0.25*9.81, 0.25*9.81])

A, B = quad.getLinearization()
Q = np.diag([10, 10, 0.1, 1, 1, 1])
R = np.diag([10, 10])
P, alpha = ComputeTerminalRegion(A,B,Q,R,umin,umax)

A, B = quad.getDiscreteLinearization()
A = scipy.sparse.csr_matrix(A)
B = scipy.sparse.csr_matrix(B)
Q = scipy.sparse.csr_matrix(Q)
R = scipy.sparse.csr_matrix(R)
N = 150

ctrl = QINF_LMPC(A, B, N, Q, R, P, alpha, xmin, xmax, umin, umax)

# simulate
simulation_time = 4
sim_N = int(simulation_time/quad.SamleRate)

X = np.empty([quad._StateDimension,sim_N])
U = np.empty([quad._InputDimension,sim_N])

x_pred = np.empty([sim_N, N+1])
y_pred = np.empty([sim_N, N+1])

time = np.arange(0.0, simulation_time, quad.SamleRate)
step = 0

X[:,step] = X0
for step,t in enumerate(time):
    u = ctrl.run(quad._state) + np.array([0.25*9.81, 0.25*9.81])  # run controler
    U[:,step] = u                        # log input
    quad.Integrate(u)                    # apply input to system
    X[:,step] = quad._state              # log the state

    x_pred[step,:] = ctrl.predictedStateTrajectory[0,:]
    y_pred[step,:] = ctrl.predictedStateTrajectory[1,:]

# visualization
x_left  = X[0,:] - np.cos(X[2,:])*0.1
x_right = X[0,:] + np.cos(X[2,:])*0.1
y_left  = X[1,:] - np.sin(X[2,:])*0.1
y_right = X[1,:] + np.sin(X[2,:])*0.1

fig1 = plt.figure(figsize=(5, 4))
ax = fig1.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')

E = ctrl.visualizeTerminalRegion()
ax.plot(E[0,:], E[1,:],'y')

lines = []
line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'g')
lines.append(line1)
lines.append(line2)

def animate(i):
    x_data = [x_left[i], x_right[i]]
    y_data = [y_left[i], y_right[i]]

    line1.set_data(x_data, y_data)
    line2.set_data(x_pred[i,:], y_pred[i,:])
    return lines

ani = animation.FuncAnimation(fig1, animate, len(X[0,:]), interval=10)

fig2, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(time, U[0,:])
ax2.plot(time, U[1,:])
ax1.set_ylabel('u1 [N]')
ax2.set_ylabel('u2 [N]')
ax2.set_xlabel('time [s]')
ax1.grid()
ax2.grid()

# fig3 = plt.figure()
# plt.plot(time, X[5,:])

plt.show()