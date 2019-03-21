import numpy as np
import svm

def mcsvmTrain(X,Y,C,kernelFunction,tol=1e-3):
    uniqueLabels = np.unique(Y)
    labelCounts = uniqueLabels.size
    #print 'multiclass SVM training'
    #print 'number of classes:',labelCounts
    model = {}
    classifiers = np.array([])
    model['uniqueLabels'] = uniqueLabels
    model['labelCounts'] = labelCounts
    classifiers = np.array([])
    #print 'Multiclass SVM classification using one-to-one strategy'
    nclassifiers = np.round(labelCounts*(labelCounts+1)/2)
    #print nclassifiers, 'SVM classifiers will be trained'
    count = 0
    for i in np.arange(labelCounts):
        for j in np.arange(i+1,labelCounts):
            label1 = uniqueLabels[i]
            label2 = uniqueLabels[j]
            count+=1
            print '{0}. SVM classification: {1:d} and {2:d}'.format(count,label1,label2)
            idx = (Y==label1) + (Y==label2)
            subX1 = X[Y==label1,:]
            subX2 = X[Y==label2,:]
            subX = np.append(subX1,subX2,axis=0)
            subY = np.append(np.ones(subX1.shape[0]),np.zeros(subX2.shape[0]))
	    m = svm.SVC(C,kernelFunction)
	    m.fit(subX,subY)
            classifier={}
            classifier['m'] = m
            classifier['idx1'] = i
            classifier['idx2'] = j
            classifiers=np.append(classifiers,classifier)
    model['classifiers'] = classifiers
    return model

def mcsvmPredict(model,X):
    m = X.shape[0]
    pred = np.zeros(m)
    preds = np.zeros((m,model['labelCounts']))
    for i in np.arange(model['classifiers'].size):
        classifier = model['classifiers'][i]
        prediction = classifier['m'].predict(X)
        preds[prediction==1,classifier['idx1']] = \
                preds[prediction==1,classifier['idx1']] + 1
        preds[prediction==0,classifier['idx2']] = \
                preds[prediction==0,classifier['idx2']] + 1
    pred = np.argmax(preds,axis=1)
    pred = model['uniqueLabels'][pred]
    return pred

