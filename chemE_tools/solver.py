import numpy as np
import scipy.optimize as opt

def x(y):
    return y**2 -1.5 

print(opt.minimize(x,1,method="nelder-mead"))