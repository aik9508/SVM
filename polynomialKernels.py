import numpy as np

def polynomialKernel(sigma,r,d):
    def f(x,y):
        return np.power(r+sigma*np.dot(x,np.transpose(y)),d)
    return f
