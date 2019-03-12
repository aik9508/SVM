from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from gaussianKernels import gaussianKernel
import multiclassSVM
import svm

m = 1000 # number of samples used for training svm classifier
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
trainImages = trainImages[:,:,:,0:m].reshape(n,m)
trainImages = np.transpose(trainImages)
trainImages = (trainImages - np.mean(trainImages,axis=0))/255
#plt.imshow(trainImages[0,:].reshape(sz[0],sz[1]))
#plt.show()
trainLabels = trainLabels[0,0:m]

ntest = 50
testImages = testImages.reshape(n,-1)
testImages = np.transpose(testImages)
testImages = (testImages - np.mean(testImages,axis=0))/255
testImages = testImages[0:ntest,:]
testLabels = testLabels[0,:]
testLabels = testLabels[0:ntest]

#Cs = [0.1,0.3,1,3,10.30]
#sigmas = [0.1,0.3,1,3,10,30]
#
#max_accuracy = 0
#C_opt = Cs[0]
#sigma_opt = sigmas[0]
#for sigma in sigmas:
#    for C in Cs:
#        kernelFunction = gaussianKernel(sigma) 
#        # training a multiclassSVM classifier
#        model = multiclassSVM.mcsvmTrain(trainImages,trainLabels,C, \
#                kernelFunction)
#        # prediction
#        pred = multiclassSVM.mcsvmPredict(model,testImages)
#        accuracy = np.mean((pred==testLabels)*1.)
#        if accuracy > max_accuracy:
#            max_accuracy = accuracy
#            sigma_opt = sigma
#            C_opt = C
#        print 'C={0:.2f}, sigma={1:.2f}, accuracy={2:.3f}'.format(C,sigma,accuracy)
#
#print 'C_opt={0:.2f}, sigma_opt={1:.2f}, max_accuracy={2:.3f}'.format(C_opt,sigma_opt,max_accuracy)

kernelFunction = gaussianKernel(sigma)
idx = (trainLabels==1) + (trainLabels==2)
subTrainImages = trainImages[idx,:]
subTrainLabels = trainLabels[idx]*1.
subTrainLabels[subTrainLabels==1]=1
subTrainLabels[subTrainLabels==2]=0


idx = (testLabels==1) + (testLabels==2)
subTestImages = testImages[idx,:]
subTestLabels = testLabels[idx]*1.
subTestLabels[subTestLabels==1]=1
subTestLabels[subTestLabels==2]=0
print subTrainImages.shape
print subTrainLabels
model = svm.svmTrain(subTrainImages,subTrainLabels,C,kernelFunction,1e-3,5)
print model
pred = svm.svmPredict(model,subTestImages)
accuracy = np.mean((pred==subTestLabels)*1.)
print accuracy
