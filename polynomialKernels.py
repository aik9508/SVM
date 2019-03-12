import numpy as np

def polynomialKernel(p):
    def f(x,y):
        return np.power(1+np.inner(x,y),p)
    return f
