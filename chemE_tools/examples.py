import numpy as np
from matplotlib import pyplot as plt
import math_funcs

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
def Cc(t):
    return 5.97435665e-03 - 2.90691266*t + 5.35489272e00*t**2 - 1.72159283e+00*t**3 + 2.26852375e-01*t**4 - 1.35633236e-02*t**5 + 3.04814898e-04*t**6
area = trapezoid(Cc,0,14,.001)
print(area)