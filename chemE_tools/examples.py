import numpy as np
from matplotlib import pyplot as plt
import math_funcs
import math

rk4 = math_funcs.ODE.rk4
rk45 = math_funcs.ODE.rk45

# # ode's governing predator prey btw rabbits and fox
# def rabbits(t,x):
#     # define variables
#     a = x[0]
#     b = x[1]

#     # set parameters
#     k1 = .02
#     k2 = .00004
#     k3 = .0004
#     k4 = .04

#     # governing equations
#     rg = k1*a - k2*a*b
#     rd = k3*a*b - k4*b

#     #gather odes
#     dx1dt = rg
#     dx2dt = rd

#     # return combined variable ode
#     dxdt = np.array([dx1dt, dx2dt])
#     return dxdt

# # ode's governing a Lorenz system
# def lorenz(t,x):
#     # define variables
#     a = x[0]
#     b = x[1]
#     c = x[2]

#     # set parameters
#     sigma = 10
#     rho = 28
#     beta = 8/3

#     #gather odes
#     dadt = sigma*(b - a)
#     dbdt = a*(rho - c) - b
#     dcdt = a*b - beta*c

#     # return combined variable ode
#     dxdt = np.array([dadt, dbdt, dcdt])
#     return dxdt

# # how to use the 4th order RK ode solver
# t1,y1 = rk4(rabbits,0,500,[500,200],.01)
# plt.plot(t1,y1,'red')
# plt.show()

# # how to use the RKF45 ode solver 
# t2,y2 = rk45(lorenz,0,35,[0,2,20],.01)
# plt.plot(y2[:,0],y2[:,2])
# plt.show()

# # how to use the definite integral function to integrate the error function
# trapezoid = math_funcs.Integration.trapezoid
# def f(x):
#     return np.exp(-x**2)
# z = 1
# area = trapezoid(f,0,z,.001)
# print((2/np.sqrt(np.pi))*area)
# print(math.erf(z))
