import svm
import numpy as np
import random
import matplotlib.pyplot as plt
from polynomialKernels import polynomialKernel

ntrain = 100
Xtrain = np.zeros((ntrain,2))
Ytrain = np.zeros(ntrain)
for i in range(ntrain):
    Xtrain[i,0] = (random.random()-0.5)*2
    Xtrain[i,1] = (random.random()-0.5)*2
    if Xtrain[i,0]*Xtrain[i,1] > 0:
        Ytrain[i]=0
    else:
        Ytrain[i]=1

ntest = 100
Xtest = np.zeros((ntest,2))
Ytest = np.zeros(ntest)
for i in range(ntest):
    Xtest[i,0] = (random.random()-0.5)*2
    Xtest[i,1] = (random.random()-0.5)*2
    if Xtest[i,0]*Xtest[i,1] > 0:
        Ytest[i]=0
    else:
        Ytest[i]=1
#A=Xtrain[Ytrain==0,:]
#B=Xtrain[Ytrain==1,:]
#plt.plot([x[0] for x in A],[x[1] for x in A],'o')
#plt.plot([x[0] for x in B],[x[1] for x in B],'o')
#plt.show()

#Xtrain = np.array([[1,-1],[3,1],[1,2],[1,-2],[-2,1],[-1,-2]])
#Ytrain = np.array([1,0,0,1,1,0])
#print Ytrain
tol = 1e-3
kf = polynomialKernel(2)
C = 1
max_passes = 5
model = svm.svmTrain(Xtrain,Ytrain,C,kf,tol,max_passes)
print model
pred = svm.svmPredict(model,Xtrain)
accuracy = np.mean((pred==Ytrain)*1.)
#print pred
#print Ytrain
print accuracy

