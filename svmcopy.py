import numpy as np
import random

def svmTrain(X,Y,C,kernelFunction,tol=1e-3,max_passes=5):
    # X,Y: training data
    # tol: numerical tolerance
    # max_passes: max number of times to iterate over alphas without changing
    # Data parameters
    m = X.shape[0]    # number of samples
    n = X.shape[1]    # dimension of a sample
    
    # Map 0 to -1
    Y[Y==0] = -1
    alphas = np.zeros(m) # Lagrange multipliers for solution
    b = 0 # threshold for solution
    E = np.zeros(m)
    K = np.zeros((m,m))
    for i in np.arange(m):
        for j in np.arange(i,m):
            K[i,j] = kernelFunction(X[i,:],X[j,:])
            K[j,i] = K[i,j]
    print K
    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + np.sum(np.multiply(np.multiply(alphas,Y),K[:,i])) - Y[i]
            if (Y[i]*E[i]<-tol) and (alphas[i]<C) or (Y[i]*E[i]>tol) and (alphas[i]>0):
                j = random.randint(0,m-1)
                while i == j:
                    j = random.randint(0,m-1)
                E[j] = b + np.sum(np.multiply(np.multiply(alphas,Y),K[:,i])) - Y[j]
                alpha_i_old = alphas[i];
                alpha_j_old = alphas[j];

                if Y[i] == Y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(0, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                if L == H:
                    continue

                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    continue

                alphas[j] = alphas[j] - Y[j]*(E[i]-E[j])/eta

                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])

                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue

                alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j])
                b1 = b - E[i] \
                        - Y[i] * (alphas[i] - alpha_i_old) * K[i,i] \
                        - Y[j] * (alphas[j] - alpha_j_old) * K[i,j] \

                b2 = b - E[j] \
                        - Y[i] * (alphas[i] - alpha_i_old) * K[i,j] \
                        - Y[j] * (alphas[j] - alpha_j_old) * K[j,j] \

                if 0 < alphas[i] and alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2

                num_changed_alphas = num_changed_alphas + 1

        if num_changed_alphas == 0:
            passes = passes + 1;
        else:
            passes = 0;

    model = {}
    idx = alphas > 0
    model['X'] = X[idx][:]
    model['Y'] = Y[idx]
    model['kernelFunction'] = kernelFunction
    model['b'] = b
    model['alphas'] = alphas[idx]
    model['w'] = np.inner(np.multiply(alphas,Y),np.transpose(X))
    return model

def svmPredict(model,X):
    X = np.array(X)
    m = X.shape[0]
    p = np.zeros(m)
    pred = np.zeros(m)
    for i in range(m):
        prediction = 0
        for j in range(model['X'].shape[0]):
            prediction = prediction + \
                    model['alphas'][j] * model['Y'][j] * \
                    model['kernelFunction'](X[i],model['X'][j])
            p[i] = prediction + model['b']
    pred[p>=0] = 1
    pred[p<0] = 0
    return pred
