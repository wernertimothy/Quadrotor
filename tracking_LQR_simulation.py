from Systems import PlanarQuadrotor
from Control import ContinuousLQR
from Trajectories import Lemniscate

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define system
quad = PlanarQuadrotor()
X0 = np.array([1, 0, 0, 0, 0, 0 ])              # initial condition
quad.set_state(X0)                              # set initial condition
quad.set_SampleRate(0.01)                       # set the sample rate

# define controler
A, B = quad.getLinearization()
ctrl = ContinuousLQR(A, B, np.diag([10, 10, 0.1, 1, 1, 1]), np.diag([10, 10]))
ctrl.setControlOffset(np.array([0.25*9.81, 0.25*9.81]))            # assuming perfect knowledge
ctrl.setBoxConstraints(np.array( [[-3, 3],                         # set min and max force
                                  [-3, 3]] ))

# define trajectory
duration = 15                   # time for on traversal
traj = Lemniscate(duration)
Lem = traj.visualize()

# simulation
simulation_time = duration
N = int(simulation_time/quad.SamleRate)

X = np.empty([quad._StateDimension,N])
U = np.empty([2,N])

time = np.arange(0.0, simulation_time, quad.SamleRate)
step = 0

X[:,step] = X0
r = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for step,t in enumerate(time):
    r[0:2] = traj.evaluate(t)
    u = ctrl.runTracking(quad._state, r) # run controler
    U[:,step] = u                        # log input
    quad.Integrate(u)                    # apply input to system
    X[:,step] = quad._state              # log the state

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

ax.plot(Lem[0,:], Lem[1,:],'r')

line, = ax.plot([], [], 'o-', lw=2)

def animate(i):
    x_data = [x_left[i], x_right[i]]
    y_data = [y_left[i], y_right[i]]

    line.set_data(x_data, y_data)
    return line,

ani = animation.FuncAnimation(fig1, animate, len(X[0,:]), interval=10)

fig2, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(time, U[0,:])
ax2.plot(time, U[1,:])
ax1.set_ylabel('u1 [N]')
ax2.set_ylabel('u2 [N]')
ax2.set_xlabel('time [s]')
ax1.grid()
ax2.grid()

plt.show()
