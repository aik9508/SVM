import numpy as np

def gaussianKernel(sigma):
    def f(x,y):
        return np.exp(-np.dot(x-y,x-y)/(2*sigma*sigma))
    return f
