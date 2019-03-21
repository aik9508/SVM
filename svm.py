import numpy as np
import random
import time

class SVC:
    def __init__(self, C, kernelFunction, tol=1e-3):
        self.C = C
        self.kernelFunction = kernelFunction
        self.tol = tol

    def fit(self,X,Y):
        t1 = time.clock()
        self.X = X
        self.y = np.copy(Y)
        self.y[Y==0] = -1
        m = Y.size    # number of samples
        self.alphas = np.zeros(m) # Lagrange multipliers
        self.E = np.zeros(m)      # error cache
        self.K = np.zeros((m,m))
        for i in np.arange(m):
            for j in np.arange(i,m):
                self.K[i,j] = self.kernelFunction(X[i,:],X[j,:])
                self.K[j,i] = self.K[i,j]
        #self.K = self.kernelFunction(X,X) 
	self.computed = np.zeros(m)
        self.b = 0        # threshold
        self.model = {}
        print time.clock()-t1

        numChanged = 0
        examineAll = True
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in np.arange(self.y.size):
                    numChanged = numChanged + self.examineExample(i)
            else:
                activeSet = np.where(np.logical_and(abs(self.alphas) \
                        > self.tol, abs(self.alphas-self.C) > self.tol))[0]
                for i in activeSet:
                    numChanged = numChanged + self.examineExample(i)
            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
        idx = self.alphas > 0
        self.model['X'] = self.X[idx,:]
        self.model['y'] = self.y[idx]
        self.model['kernelFunction'] = self.kernelFunction
        self.model['b'] = self.b
        self.model['alphas'] = self.alphas[idx]
        self.model['w'] = np.inner(self.alphas*self.y,np.transpose(self.X))
	#print np.mean(self.computed)
        self.X = []
        self.y = []
        self.K = []
        self.E = []
        self.alphas = []
        print time.clock()-t1

    def takeStep(self,i,j):
        if i==j:
            return 0
        m = self.X.shape[0]
	#if self.computed[i] == 0:
	#    for k in np.arange(m):
	#	self.K[i,k] = self.kernelFunction(self.X[i,:],self.X[k,:])
	#    self.computed[i] = 1
	#if self.computed[j] == 0:
	#    for k in np.arange(m):
	#	self.K[j,k] = self.kernelFunction(self.X[i,:],self.X[k,:])
	#    self.computed[j] = 1
        #Ki = self.K[i,:]
        #Kj = self.K[j,:]
	self.E[i] = self.b + np.sum(self.alphas*self.y*self.K[i,:]) - self.y[i]
	self.E[j] = self.b + np.sum(self.alphas*self.y*self.K[j,:]) - self.y[j]
        alpha_i_old = self.alphas[i];
        alpha_j_old = self.alphas[j];
        if self.y[i] == self.y[j]:
            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        else:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        if L == H:
            return 0

        #Kij = self.kernelFunction(self.X[i,:],self.X[j,:])
        #Kii = self.kernelFunction(self.X[i,:],self.X[i,:])
        #Kjj = self.kernelFunction(self.X[j,:],self.X[j,:])
        Kij = self.K[i,j]
        Kii = self.K[i,i]
        Kjj = self.K[j,j]
        
        # second order derivative of the cost function with respect to
        # alphas[i] and alphas[j] along the constraint segment
        eta = 2 * Kij - Kii - Kjj
        if eta >= 0:
            return 0

        # update Lagrange multipliers
        self.alphas[j] = self.alphas[j] - self.y[j]*(self.E[i]-self.E[j])/eta
        self.alphas[j] = min(H, self.alphas[j])
        self.alphas[j] = max(L, self.alphas[j])
        if abs(self.alphas[j] - alpha_j_old) < self.tol:
            self.alphas[j] = alpha_j_old
            return 0
        self.alphas[i] = self.alphas[i] + self.y[i]*self.y[j]*(alpha_j_old - self.alphas[j])
        
        # update threshold to reflect change in Lagrange multipliers
        b1 = self.b - self.E[i] \
                - self.y[i] * (self.alphas[i] - alpha_i_old) * Kii \
                - self.y[j] * (self.alphas[j] - alpha_j_old) * Kij
        b2 = self.b - self.E[j] \
                - self.y[i] * (self.alphas[i] - alpha_i_old) * Kij \
                - self.y[j] * (self.alphas[j] - alpha_j_old) * Kjj
        if 0 < self.alphas[i] and self.alphas[i] < self.C:
            self.b = b1
        elif 0 < self.alphas[j] and self.alphas[j] < self.C:
            self.b = b2
        else:
            self.b = (b1+b2)/2
        return 1
    
    def examineExample(self,i):
        #m = self.y.size
        #Ki = np.zeros(m)
        #for k in np.arange(m):
        #    Ki[k] = self.kernelFunction(self.X[i,:],self.X[k,:])
        Ki = self.K[i,:]
        self.E[i] = self.b + np.sum(self.alphas*self.y*Ki) - self.y[i]
        if (self.y[i]*self.E[i]<-self.tol) and (self.alphas[i]<self.C) \
                or (self.y[i]*self.E[i]>self.tol) and (self.alphas[i]>0):
            activeSet = np.where(np.logical_and(abs(self.alphas) > self.tol, abs(self.alphas-self.C) > self.tol))[0]
            m = activeSet.size
            if m > 1:
                if self.E[i] > 0:
                    j = activeSet[np.argmin(self.E[activeSet])]
                else:
                    j = activeSet[np.argmax(self.E[activeSet])]
                if self.takeStep(j,i) > 0:
                    return 1
            if m > 0:
                r = random.randint(0,m-1)
                activeSet = np.concatenate([activeSet[r:m],activeSet[0:r]])
                for j in np.arange(m):
                    if self.takeStep(activeSet[j],i) > 0:
                        return 1
            r = random.randint(0,self.y.size-1)
            idx = np.concatenate([np.arange(r,self.y.size),np.arange(0,r)])
            for j in idx:
                if self.takeStep(j,i) > 0:
                    return 1
        return 0
    
    def predict(self,X):
        max_samples = 3000
        m = X.shape[0]
        p = np.zeros(m)
        pred = np.zeros(m)
        #p = np.dot(self.model['alphas']*self.model['y'], \
        #        self.model['kernelFunction'](self.model['X'],X)) + self.model['b']
        for i in range(m):
            prediction = 0
            for j in range(self.model['X'].shape[0]):
                prediction = prediction + \
                        self.model['alphas'][j] * self.model['y'][j] * \
                        self.model['kernelFunction'](X[i,:],self.model['X'][j,:])
            p[i] = prediction + self.model['b']
        pred[p>=0] = 1
        return pred
