import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class Regression:
    def get_mult_linreg(x_train: npt.NDArray, y_train: npt.NDArray, order: int, x_test: npt.NDArray = False) -> npt.NDArray:
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
                mult_linreg_mdl = Betas[0] + Betas[1]*x_test
            else:
                mult_linreg_mdl = Betas[0]
                for j in range(1,order+1):
                    mult_linreg_mdl = mult_linreg_mdl + Betas[j]*(x_test**j)
            return mult_linreg_mdl, Betas

    def get_r2(x: npt.NDArray, y: npt.NDArray) -> float:
        yhat = np.mean(y)
        pred,betas = Regression.get_mult_linreg(x,y,1)
        RSS = np.sum(np.square(y-pred))
        TSS = np.sum(np.square(y-yhat))
        return 1 - RSS/TSS

    def get_adj_r2(x: npt.NDArray, y: npt.NDArray, n, d) -> float:
        yhat,betas = Regression.get_mult_linreg(x,y,1)
        ybar = np.mean(y)
        RSS = np.sum(np.square((y - yhat)))
        TSS = np.sum(np.square((y - ybar)))
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