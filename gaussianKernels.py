import numpy as np

def gaussianKernel(sigma):
    def f(x,y):
        return np.exp(-np.dot(x-y,x-y)/(2*sigma**2))
        #if np.ndim(x) == 1 and np.ndim(y) == 1:
        #    return np.exp(-np.linalg.norm(x-y)/(2*sigma**2))
        #elif (np.ndim(x) > 1 and np.ndim(y) == 1) or \
        #        (np.ndim(x) == 1 and np.ndim(y) > 1):
        #    return np.exp(-np.linalg.norm(x-y, axis=1)/(2*sigma**2))
        #else:
        #    return np.exp(-np.linalg.norm(x[:,np.newaxis]-y[np.newaxis,:], \
        #            axis=2) / (2*sigma**2))
    return f
