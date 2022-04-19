import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def random_walk(n_steps=1000,step_size=1,x_min=0,x_max=10,y_min=0,y_max=10,runs=1000):
    r = np.zeros([n_steps,2])
    r[0,:] = [(x_min+x_max)/2,(y_min+y_max)/2]
    trial = np.zeros([runs,1])
    for j in tqdm(range(0,runs)):
        for i in range (1,n_steps):
            xstep = np.random.uniform(-step_size, step_size)
            ystep = np.random.uniform(-step_size, step_size)
            x = r[i-1,0]
            y = r[i-1,1]

            if x + xstep < x_min or x+xstep > x_max:
                x += -xstep
            if y + ystep > y_max or y+ystep <y_min: 
                y += -ystep
            else: 
                x += xstep
                y += ystep
            steps = i
            r[i,0] = x
            r[i,1] = y
        trial[j] = steps

    l = np.linspace(0, steps+1, steps+1)
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    plt.scatter(r[:,0], r[:, 1], c=l, cmap='winter')
    return r
# random_walk(1000,1,0,20,0,20,1)