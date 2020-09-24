import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

Q = np.matrix([
    [0, 4],
    [1, 1]
])
D = np.matrix(np.diag([0.5, 1.5]))

alpha = 0.6

P = Q*D*scipy.linalg.inv(Q)
H = alpha*scipy.linalg.inv(P)
T = scipy.linalg.sqrtm(H)

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

g = 100
test_x = np.linspace(-2, 2, g)
test_y = test_x

for k in range(0,g):
    for l in range(0,g):
        v = np.array([test_x[k], test_y[l]])
        if v.T@P@v <= alpha:
            plt.plot(v[0], v[1], '.g')
        else:
            plt.plot(v[0], v[1], '.r')

plt.plot(x,y)
plt.plot(E[0,:], E[1,:])

plt.axes().set_aspect('equal')
plt.show()