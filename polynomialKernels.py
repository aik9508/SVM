import numpy as np

def polynomialKernel(p):
    def f(x,y):
        return np.power(1+np.dot(x,np.transpose(y)),p)
    return f
