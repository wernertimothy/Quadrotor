import casadi
import numpy as np
import matplotlib.pyplot as plt

# === define parameter ===
nx = 6 # state dim
nu = 2 # input dim

m = 0.5
I = 0.002
l = 0.1
g = 9.81
p = np.array([m, I, l, g])

dt = 0.01 # samplerate
N  = 50   # discrete horizon

# === create ode expression ===

x = casadi.MX.sym('x', nx)
u = casadi.MX.sym('u', nu)
# p = casadi.MX.sym('p', np.size(parameter))

# ode = casadi.vertcat(
#     x[3]                                         ,\
#     x[4]                                         ,\
#     x[5]                                         ,\
#     -1/p[0]*(u[0] + u[1])*casadi.sin(x[2])       ,\
#      1/p[0]*(u[0] + u[1])*casadi.cos(x[2]) - p[3],\
#     p[2]/p[1]*(u[0] - u[1])
# )
ode = casadi.vertcat(
    x[3]                                   ,\
    x[4]                                   ,\
    x[5]                                   ,\
    -1/m*(u[0] + u[1])*casadi.sin(x[2])    ,\
     1/m*(u[0] + u[1])*casadi.cos(x[2]) - g,\
    l/I*(u[0] - u[1])
)

f = casadi.Function('f', [x,u], [ode], ['x','u'], ['ode'])

# === create integrator expression ===
intg_options = dict(tf = dt)            # define integration step
dae = dict(x = x, p = u, ode = f(x,u))  # define the dae to integrate (ZOH)

# define the integrator
intg = casadi.integrator('intg', 'rk', dae, intg_options) 
res = intg(x0 = x, p = u)
# define symbolic expression of integrator
x_next = res['xf']  
# define discretized ode
F = casadi.Function('F', [x, u], [x_next], ['x', 'u'], ['x_next']) 

# === define OCP ===
'''
minimize      sum_{k=0}N^{N-1} ||x_k||_Q + ||u_k||_R + ||x_N||_P
subject to    x_0   = x_t
              x_k+1 = F(x_k, u_k)
              _x <= x_k <= x_
              _u <= u_k <= u_
'''

Q = casadi.MX.sym('Q', nx, nx)
R = casadi.MX.sym('R', nu, nu)

qsum_x = casadi.Function('qsum_x', [x,Q], [casadi.bilin(Q,x,x)], ['x','Q'], ['xQx'])
qsum_u = casadi.Function('qsum_u', [u,R], [casadi.bilin(R,u,u)], ['u','R'], ['uRu'])

Q = np.diag([10.0, 10.0, 1.0, 1.0, 1.0, 0.1])
R = np.diag([1.0, 1.0])
P = 10*Q

umin = np.array([0.5,0.5])
umax = np.array([3.0, 3.0])

opti = casadi.Opti()
x  = opti.variable(nx, N+1)
u  = opti.variable(nu, N)
x0 = opti.parameter(nx,1)

# initial condition
opti.subject_to( x[:,0] == x0 )

# stage 1:N
objective = 0
for k in range(0,N):
    objective += qsum_x(x[:,k], Q) + qsum_u(u[:,k], R) # stage cost
    opti.subject_to( x[:,k+1] == F(x[:,k], u[:,k]) )   # dynamics
    # input constraints
    opti.subject_to( umin <= u[:,k] )
    opti.subject_to( u[:,k] <= umax )
# stage N+1
objective += qsum_x(x[:,N], P) # terminal cost

opti.minimize(objective)

# set solver
opti.solver('sqpmethod', dict(qpsol = 'osqp'))
opti.solver('ipopt')

# set initial condition
X0 = np.array([-0.3, -0.8, 0.6, 0.1, -0.8, 0.2 ])
opti.set_value(x0, X0)

# solve OCP
sol = opti.solve()


# === visualization ===

time = np.arange(0,N*dt, dt)
plt.plot(sol.value(x[0,:]), sol.value(x[1,:]))

fig2, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(time, sol.value(u[0,:]))
ax2.plot(time, sol.value(u[1,:]))
ax1.set_ylabel('u1 [N]')
ax2.set_ylabel('u2 [N]')
ax2.set_xlabel('time [s]')
ax1.grid()
ax2.grid()

plt.show()
