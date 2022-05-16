from tqdm import tqdm
import numpy as np
import numpy.typing as npt

class Regression:
    def get_poly_linreg(x_train: npt.NDArray, y_train: npt.NDArray, order: int, x_test: npt.NDArray = False) -> npt.NDArray:
            if isinstance(x_test,bool) == True:
                x_test = x_train
            Regression.checknan(x_train)
            Regression.checknan(x_train)
            Regression.checknan(x_test)
            y = np.reshape(y_train,[len(y_train),1])
            x = np.reshape(x_train,[len(x_train),1])
            if order == 1:
                x_ones = np.ones([len(x_train),1])
                x = np.hstack((x_ones,x))
            else: 
                x_ones = np.ones([len(x_train),1])
                for i in range(1,order+1):
                    x_ones = np.hstack((x_ones,x**i))
                x = x_ones
            Betas = np.dot(np.linalg.inv(np.dot(x.transpose(1,0),x)),np.dot(x.transpose(1,0),y))
            if order == 1:
                poly_linreg_mdl = Betas[0] + Betas[1]*x_test
            else:
                poly_linreg_mdl = Betas[0]
                for j in range(1,order+1):
                    poly_linreg_mdl = poly_linreg_mdl + Betas[j]*(x_test**j)
            return poly_linreg_mdl, Betas

    def get_mult_linreg(x_train: npt.NDArray, y_train: npt.NDArray, x_test: npt.NDArray = False) -> npt.NDArray:
            if isinstance(x_test,bool) == True:
                x_test = x_train
            Regression.checknan(x_train)
            Regression.checknan(x_train)
            Regression.checknan(x_test)
            y = np.reshape(y_train,[len(y_train),1])
            x = np.reshape(x_train,[len(x_train[:,0]),len(x_train[0,:])])
            x_ones = np.ones([len(x_train[:,0]),1])
            x = np.hstack((x_ones,x))
            Betas = np.dot(np.linalg.inv(np.dot(x.transpose(1,0),x)),np.dot(x.transpose(1,0),y))
            mult_linreg_mdl = np.ones([len(x_test[:,0])])*Betas[0]
            for j in range(0,len(x_test[0,:])):
                mult_linreg_mdl = mult_linreg_mdl + Betas[j+1]*(x_test[:,j])
            return mult_linreg_mdl, Betas

    def get_r2_regress(x_train: npt.NDArray, y_train: npt.NDArray, x_test: npt.NDArray = False, y_test: npt.NDArray = False) -> float:
        if isinstance(x_test,bool) == True:
                x_test = x_train
        if isinstance(y_test,bool) == True:
                y_test = y_train
        yhat = np.mean(y_test)
        pred,betas = Regression.get_poly_linreg(x_train,y_train,1,x_test)
        RSS = np.sum(np.square(y_test-pred))
        TSS = np.sum(np.square(y_test-yhat))
        return 1 - RSS/TSS

    def get_adj_r2_regress(x_train: npt.NDArray, y_train: npt.NDArray, n: int, d: int, x_test: npt.NDArray = False, y_test: npt.NDArray = False) -> float:
        if isinstance(x_test,bool) == True:
                x_test = x_train
        if isinstance(y_test,bool) == True:
                y_test = y_train
        yhat,betas = Regression.get_poly_linreg(x_train,y_train,1,x_test)
        ybar = np.mean(y_test)
        RSS = np.sum(np.square((y_test - yhat)))
        TSS = np.sum(np.square((y_test - ybar)))
        return 1-((RSS/(n-d-1))/(TSS/(n-1)))

    def checknan(array: npt.NDArray,name: str = "your stinky data") -> bool:
        k = len(np.shape(array))
        flag = False
        if k == 1:
            r = np.shape(array)[0]
            nan_bool = np.isnan(array)
            for i in range(0,r):
                if nan_bool[i]:
                    flag = True
                    print("warning: NaN detected at ["+str(i)+"] in " + name)
        elif k == 2:
            r = np.shape(array)[0]
            c = np.shape(array)[1]
            nan_bool = np.isnan(array)
            for i in range(0,r):
                for j in range(0,c):
                    if nan_bool[i,j]:
                        flag = True
                        print("warning: NaN detected at [" + str(i) + "," +str(j)+"] in " + name)
        return flag

class Stats:
    def r2(y_test: npt.NDArray, pred: npt.NDArray) -> float:
        ybar = np.mean(y_test)
        ss_res = np.sum(np.square(y_test-pred))
        ss_tot = np.sum(np.square(y_test-ybar))
        return 1 - (ss_res/ss_tot)

    def adj_r2(y_test: npt.NDArray, pred: npt.NDArray, n: int, p: int) -> float:
        return 1 - ((1-Stats.r2(y_test,pred))*(n-1)/(n-p))

class ODE:
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

    def rk45(f,t0,tf,ic,step_size):
        ic = np.array([ic])
        fourth_approx = np.array([])
        fifth_approx = np.array([])
        tspan = np.array([t0, tf])
        h = np.array([])
        t = np.array([])
        y = np.array(ic)
        h = step_size
        t = np.append(t,tspan[0])
        tol = .00001
        i = 0
        while t[-1]<tf:
            k1 = h*f(t[i], y[i])
            k2 = h*f(t[i] + h/4, y[i] + k1/4)
            k3 = h*f(t[i] + 3*h/8, y[i] + 3*k1/32 + 9*k2/32)
            k4 = h*f(t[i] + 12*h/13, y[i] + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            k5 = h*f(t[i] + h, y[i] + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
            k6 = h*f(t[i] + h/2, y[i] - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
            fourth_approx = y[i] + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
            fifth_approx = y[i] + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55

            R = np.abs(np.array(fifth_approx)-np.array(fourth_approx))/h
            R = np.amax(R)
            delta = (tol/(2*R))**(1/4)

            if R<=tol:
                t = np.append(t, t[-1] + h)
                y = np.append(y, np.array([fourth_approx]), axis=0)
                if delta < 0.1:
                    h = 0.1*h
                elif delta > 4:
                    h = 4*h
                else:
                    h = delta*h
                i+=1
            else:
                h = delta*h
        print("steps:",i+1)
        return t,y

class Integration:
    def trapezoid(function,lower_limit,upper_limit,step_size=.01):
        area = np.zeros(1)
        for i in tqdm(np.arange(lower_limit,upper_limit,step_size)):
            area = np.append(area,(step_size)*((function(i)+function(i+step_size))/2))
        area = sum(area)
        return area
