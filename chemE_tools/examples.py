import numpy as np
from matplotlib import pyplot as plt
import math_funcs
import math

rk4 = math_funcs.ODE.rk4
def rabbits(t,x):
    # define variables
    a = x[0]
    b = x[1]

    # set parameters
    k1 = .02
    k2 = .00004
    k3 = .0004
    k4 = .04

    # governing equations
    rg = k1*a - k2*a*b
    rd = k3*a*b - k4*b

    #gather odes
    dx1dt = rg
    dx2dt = rd

    # return combined variable ode
    dxdt = np.array([dx1dt, dx2dt])
    return dxdt

t1,y1 = rk4(rabbits,0,500,[500,200],.01)
plt.plot(t1,y1)
plt.show()

trapezoid = math_funcs.Integration.trapezoid

def f(x):
    return np.exp(-x**2)

z = 1
area = trapezoid(f,0,z,.001)
print((2/np.sqrt(np.pi))*area)
print(math.erf(z))
