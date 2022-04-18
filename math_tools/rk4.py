import numpy as np
from tqdm import tqdm

def rk4(f,t0,tf,ic,step_size):
    num_eqn = int(len(ic))
    h = step_size
    tspan = np.array([t0, tf])
    n = int((tspan[1]-tspan[0])/h)
    y = np.zeros((n,num_eqn))
    t = np.zeros(n)
    t[0] = tspan[0]
    for j in range(0,num_eqn):
        y[0,j] = ic[j]
    for i in tqdm(range(0,n-1)):
        k1 = f(t[i],y[i,:])
        k2 = f(t[i]+h/2,y[i,:]+h*k1/2)
        k3 = f(t[i]+h/2,y[i,:]+h*k2/2)
        k4 = f(t[i]+h,y[i,:]+h*k3)
        y[i+1,:] = y[i,:] + 1/6*h*(k1+2*k2+2*k3+k4)
        t[i+1] = t[i]+h
    return t,y