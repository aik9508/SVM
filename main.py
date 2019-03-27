from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time
from gaussianKernels import gaussianKernel
from polynomialKernels import polynomialKernel
import multiclassSVM

f = open('multiclass_poly.txt','w')
ntrains = np.append(100*np.arange(1,10),1000*np.arange(1,11))
ntest = 100
C = 1 
sigma = 1 # standard deviation for the Gaussian Kernel
d = 1
r = 1
# number of samples used for training svm classifier
for ntrain in ntrains:
    digits =loadmat('digits.mat')
    trainLabels = digits['trainLabels']
    trainImages = digits['trainImages']
    testImages = digits['testImages']
    testLabels = digits['testLabels']
    sz = trainImages.shape
    n = sz[0]*sz[1] # pixels of an digital image

    # preprocessing training and test data
    trainImages = trainImages[:,:,:,0:ntrain].reshape(n,ntrain)
    trainImages = np.transpose(trainImages)
    trainImages = (trainImages - np.mean(trainImages,axis=0))/255
    trainLabels = trainLabels[0,0:ntrain]

    testImages = testImages.reshape(n,-1)
    testImages = np.transpose(testImages)
    testImages = (testImages - np.mean(testImages,axis=0))/255
    testImages = testImages[-ntest:,:]
    testLabels = testLabels[0,:]
    testLabels = testLabels[-ntest:]

    #kernelFunction = gaussianKernel(sigma) 
    kernelFunction = polynomialKernel(sigma,r,d)
    # training a multiclassSVM classifier
    model = multiclassSVM.mcsvmTrain(trainImages,trainLabels,C, \
            kernelFunction)
    # prediction
    pred = multiclassSVM.mcsvmPredict(model,testImages)
    accuracy = np.mean((pred==testLabels)*1.)
    print 'ntrain:{0:d}, accuracy:{1:f}'.format(ntrain,accuracy)
    f.write('ntrain: %d, accuracy: %f\n'%(ntrain,accuracy))
f.close()
