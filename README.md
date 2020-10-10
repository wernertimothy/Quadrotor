# Quadrotor
simulation of different controlers for quadrotor stabilization, path following and reference tracking

## ToDo
* make Control.ComputeTerminalRegion run
* add Jacobian and Hessian to QINF NLP
* add LTV tracking
* make scipy.sparse.csr_matrix() inside the controller
* add meassuerement/noise simulation
* add estimation (Kalman, EKF)

## Dependencies
To run the code the following packages are required:
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [cvxpy](https://www.cvxpy.org/index.html)
* [CasADi](https://web.casadi.org/)

to install them use

```python
# Linux
python3 -m pip install numpy scipy matplotlib cvxpy casadi
```

or see the documentation on the website. I recommend using [VS Code](https://code.visualstudio.com/) and to work inside a [virtual environment](https://code.visualstudio.com/docs/python/python-tutorial). To enable gloabl packages within the environment in .venv/pyvenv.cfg set

```python
include-system-site-packages = true
```

## Systems
The systems are defined in the *Systems* script. The following are the implementes systems:

* **Planar Quadrotor**. See e.g. [here](http://underactuated.mit.edu/acrobot.html#section3)

## Control
The controlers are defined in the *Control* script. The following are the implemented controlers:
### LQR
Solution to the infinite horizon quadtratic cost optimal control problem with linear dynamics. As a reference see e.g. [this](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) of for a more in depth discussion see [this](https://www.acin.tuwien.ac.at/file/teaching/master/Regelungssysteme-1/VO_Regelungssysteme1_2018.pdf).

As an example see

```python
stabilizing_LQR_simulation.py # stabilizes the system at the origin
tracking_LQR_simulation.py    # tracks a trajectory
```


### Zero Terminal Constraint (ZTC) LMPC
Solution to the finite horizon quadtratic cost optimal control problem with linear dynamics and additional state and input constraints. To enforce stability the terminal state is constraint to be zero (hence the name ZTC).

As an example see

```python
ZTC_LMPC_simulation.py # stabilizes the system at the origin
```

### Quasi Infinit Horizon (QINF) LMPC
Solution to the finite horizon quadtratic cost optimal control problem with linear dynamics and additional state and input constraints. To enforce stability the terminal state is constraint to a terminal region and the stage cost is extended with a terminal cost. This terminal cost has the value of the LQR problem with the terminal state as initial condition and thus approximates an infinite horizon (hence the name QINF. As a reference see [Chen, Allgöwer 1998](http://www.paper.edu.cn/scholar/showpdf/OUD2INwINTj0IxeQh).

As an example see

```python
QINF_LMPC_simulation.py # stabilizes the system at the origin
```

### Output Tracking LMPC
For a discussion of the problem see [here](https://github.com/wernertimothy/Quadrotor/blob/master/doc/Linear_Tracking_MPC.pdf).

As an example see

```python
tracking_LMPC_simulation.py # tracks the lemnsicate with rate constraints
```

<img src="https://github.com/wernertimothy/Quadrotor/blob/master/doc/tracking_LMPC.gif" />
<img src="https://github.com/wernertimothy/Quadrotor/blob/master/doc/tracking_LMPC_input.png" />