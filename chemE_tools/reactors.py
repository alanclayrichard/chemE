import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import ttk
import math_funcs as m
rk45 = m.ODE.rk45

def PFR(V,F):
    # initialize variables (flow rates)
    Fa = F[0]
    Fb = F[1]
    Fc = F[2]
    Fd = F[3]
    Fe = F[4]
    Ff = F[5]
    Fg = F[6]
    T = F[7]

    # set system parameters
    Pt0 = 2.4
    Cpa = 299
    Cpb = 273
    Cpc = 30
    Cpd = 201
    Cpe = 90
    Cpf = 249
    Cpg = 68
    Cpi = 40
    dH1 = 118000
    dH2 = 105200
    dH3 = -53900
    b1 = -17.34
    b2 = -1.302e4
    b3 = 5.051
    b4 = -2.314e-10
    b5 = 1.302e-6
    b6 = -4.931e-3
    rho = 2137
    phi = 0.4
    theta = 14.5

    # set governing equations
    Fi = theta*.00344
    Kp1 = np.exp(b1 + b2/T + b3*np.log(T) + ((b4*T+b5)*T+b6)*T)
    FT = Fa + Fb + Fc + Fd + Fe + Ff + Fg + Fi

    # stoichiometry 
    Pa = Fa/FT*Pt0
    Pb = Fb/FT*Pt0
    Pc = Fc/FT*Pt0

    # set governing reactions
    r1 = rho*(1-phi)*np.exp(-0.08539 - 10925/T)*(Pa - Pb*Pc/Kp1)
    r2 = rho*(1-phi)*np.exp(13.2392 - 25000/T)*(Pa)
    r3 = rho*(1-phi)*np.exp(0.2961 - 11000/T)*(Pa*Pc)
    ra = -r1 - r2 - r3
    rb = r1
    rc = r1 - r3
    rd = r2
    re = r2
    rf = r3
    rg = r3

    # mol and nrg balances
    dFadV = ra
    dFbdV = rb
    dFcdV = rc
    dFddV = rd
    dFedV = re
    dFfdV = rf
    dFgdV = rg
    Qr = 0
    Qg = r1*dH1 + r2*dH2 + r2*dH3
    sumofFCp = Fa*Cpa + Fb*Cpb + Fc*Cpc + Fd*Cpd + Fe*Cpe + Ff*Cpf + Fg*Cpg + Fi*Cpi
    dTdV = -(Qg - Qr)/sumofFCp
    
    # gather ODE's
    dFdV = np.array([dFadV, dFbdV, dFcdV, dFddV, dFedV, dFfdV, dFgdV, dTdV])
    return dFdV

def CSTR():
    print('you usgly')
    root.destroy()

def solve_PFR():
    # Solve with rk45 from math_funcs
    Vol,Flows = rk45(PFR,0,10,[.0034, 0, 0, 0, 0, 0, 0, 930],.01)
    # plot flow rates as function of volume
    plt.plot(Vol,Flows[:,0])
    plt.plot(Vol,Flows[:,1])
    plt.plot(Vol,Flows[:,2])
    plt.plot(Vol,Flows[:,3])
    plt.plot(Vol,Flows[:,4])
    plt.plot(Vol,Flows[:,5])
    plt.plot(Vol,Flows[:,6])
    plt.show()
    root.destroy()

root = Tk()
root.title('Reactor Solutions')
root.eval('tk::PlaceWindow . center')
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Choose Reactor Type: ").grid(column=0, row=0)
ttk.Button(frm, text="PFR", command=solve_PFR).grid(column=1, row=0)
ttk.Button(frm, text="CSTR", command=CSTR).grid(column=1, row=1)
root.mainloop()