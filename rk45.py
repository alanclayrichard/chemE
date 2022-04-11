import numpy as np
from tqdm import tqdm

def rk45(f,t0,tf,ic,step_size):
    ic = np.array([ic])

    fourth_approx = np.array([])
    fifth_approx = np.array([])

    tspan = np.array([t0, tf])

    h = np.array([])
    t = np.array([])
    y = np.array([[0,0]])

    h = np.append(h,step_size)
    t = np.append(t,tspan[0])
    y = np.append(y,ic,axis=0)

    tol = .00001
    i = 0

    while t[-1] < tspan[1]:
        flag = False
        current_step = h[i]
        while flag == False:
            k1 = current_step*f(t[i],y[i+1])
            k2 = current_step*f(t[i]+current_step/4,y[i+1]+k1/4)
            k3 = current_step*f(t[i]+3*current_step/8,y[i+1]+3*k1/32+9*k2/32)
            k4 = current_step*f(t[i]+12*current_step/13,y[i+1]+1932*k1/2197-7000*k2/2197+7296*k3/2197)
            k5 = current_step*f(t[i]+current_step,y[i+1]+439*k1/216-8*k2+3680*k3/513-845*k4/4104)
            k6 = current_step*f(t[i]+current_step/2,y[i+1]-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40)
            fourth_approx = y[i+1] + 25*k1/216+1408*k3/2565+2197*k4/4101-k5/5
            fifth_approx = y[i+1] + 16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55

            R = 1/np.array(h[i])*np.abs(np.array(fifth_approx)-np.array(fourth_approx))
            delta = (tol/(2*R))**(1/4)
            print(R)
            if R.all() <= tol:
                y = np.append(y,fourth_approx,axis=0)
                # print(h)
                h = np.append(h,delta*h[i])
                # print(h)
                t = np.append(t,(t[i]+h[i+1]))
                flag = True
            else:
                current_step = delta*h[i]
                flag = False
        i += 1
    return t,y