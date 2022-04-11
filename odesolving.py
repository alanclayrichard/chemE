import numpy as np
from matplotlib import pyplot as plt
from rk4 import rk4
from rk45 import rk45

def algae(t,x):
    # define variables
    Cc = x[0]

    # set parameters
    mu = 0.9/24
    t = np.mod(t,24)
    print(type(t))
    if t.all() < 12:
        f = np.sin(np.pi*t/12)
    else:
        f=0

    # governing equations
    rg = f*mu*Cc

    #gather odes
    dCdt = rg

    # return combined variable ode
    dxdt = np.array([dCdt])
    return dxdt

# t1,y1 = rk4(algae,0,48,[1],.001)
t2,y2 = rk45(algae,0,48,[1, 2],.001)
# plt.plot(t1,y1)
plt.plot(t2,y2)
plt.show()