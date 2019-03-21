from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time
from gaussianKernels import gaussianKernel
from polynomialKernels import polynomialKernel
import multiclassSVM

f = open('multiclass_poly.txt','w')
ntrains = np.append(100*np.arange(1,10),1000*np.arange(1,11))
# number of samples used for training svm classifier
for ntrain in ntrains:
    ti = time.clock()
    C = 1 
    sigma = 5 # standard deviation for the Gaussian Kernel
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
    #plt.imshow(trainImages[0,:].reshape(sz[0],sz[1]))
    #plt.show()
    trainLabels = trainLabels[0,0:ntrain]

    ntest = 10000
    testImages = testImages.reshape(n,-1)
    testImages = np.transpose(testImages)
    testImages = (testImages - np.mean(testImages,axis=0))/255
    testImages = testImages[-ntest:,:]
    testLabels = testLabels[0,:]
    testLabels = testLabels[-ntest:]

    Cs = [1]
    sigmas = [5]

    max_accuracy = 0
    C_opt = Cs[0]
    sigma_opt = sigmas[0]
    for sigma in sigmas:
	for C in Cs:
	    #kernelFunction = gaussianKernel(sigma) 
	    kernelFunction = polynomialKernel(1)
	    # training a multiclassSVM classifier
	    model = multiclassSVM.mcsvmTrain(trainImages,trainLabels,C, \
		    kernelFunction)
	    tf1 = time.clock()
	    print 'time for training', tf1-ti, 's'
	    # prediction
	    pred = multiclassSVM.mcsvmPredict(model,testImages)
	    print 'time for predicting', time.clock()-tf1, 's'
	    print 'total elapsed time', time.clock()-ti, 's'
	    accuracy = np.mean((pred==testLabels)*1.)
	    if accuracy > max_accuracy:
		max_accuracy = accuracy
		sigma_opt = sigma
		C_opt = C
	    print 'C={0:.2f}, sigma={1:.2f}, accuracy={2:.3f}'.format(C,sigma,accuracy)

    print 'C_opt={0:.2f}, sigma_opt={1:.2f}, max_accuracy={2:.3f}'.format(C_opt,sigma_opt,max_accuracy)
    f.write('ntrain: %d, accuracy: %f\n'%(ntrain,accuracy))

f.close()
