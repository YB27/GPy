
''' temporary : Use this to use the forked version of GPy'''
import sys
sys.path.insert(1, '/home/breux/GPy')

import numpy as np
import matplotlib.pyplot as plt
import GPy
'''
Gaussian Processes regression with censored data example using artificial data as in 
"Gaussian Process Regression with Censored Data Using Expectation Propagation, P. Groot, P. Lucas"  
'''

def f(x):
    return ((6.*x - 2.)**2)*np.sin(2.*(6.*x-2.))

def plotModels(m_tobit, m_normal, m_normalWithoutCensoredData):
    plt.title("Tobit GP model")
    m_tobit.plot(fignum=0)

    plt.title("Standart GP model")
    m_normal.plot(fignum=1)

    plt.figure(2)
    plt.title("Standart GP model without censored data")
    m_normalWithoutCensoredData.plot(fignum=2)

def artificialExample():
    ''' Generate Data '''
    n = 30
    x = np.linspace(0,1,n)
    y = f(x) + np.random.normal(0, np.sqrt(0.2), x.shape[0])
    x = x.reshape((n,1))
    l = -0.2265
    lowerCensoredData = np.zeros((n,), dtype='int64')
    lowerCensoredData_indexes = [idx for idx, val in np.ndenumerate(y) if val < l]
    np.put(lowerCensoredData, lowerCensoredData_indexes, 1)
    gaussianData = np.zeros((n,), dtype='int64')
    gaussianData_indexes = [idx for idx, val in np.ndenumerate(y) if val >= l]
    np.put(gaussianData, gaussianData_indexes, 1)

    y_metadata = {"lowerCensored": lowerCensoredData.reshape((n,1)), "gaussianIndexes": gaussianData.reshape((n,1))}
    #y_metadata = {"lowerCensored": np.array([idx for idx, val in np.ndenumerate(y) if val < l]),
    #              "gaussianIndexes": np.array([idx for idx, val in np.ndenumerate(y) if val >= l])}

    ''' Censored data '''
    yc = y.copy()
    np.put(yc, y_metadata["lowerCensored"], l)

    ''' Data without censored data'''
    yc2 = np.delete(yc, lowerCensoredData_indexes)
    x2 = np.delete(x,lowerCensoredData_indexes)
    yc = yc.reshape((n,1))
    yc2 = yc2.reshape(yc2.shape[0],1)
    x2 = x2.reshape(x2.shape[0],1)


    ''' GP models '''
    y = y.reshape((n,1))
    kernel = GPy.kern.RBF(input_dim=1, variance=5, lengthscale=0.1)
    m_tobit = GPy.models.GPRegressionCensored(x, yc, lowerThreshold=l, upperThreshold=None, kernel=kernel, Y_metadata=y_metadata)
    m_normal = GPy.models.GPRegression(x, y, kernel=kernel)
    m_normalWithoutCensoredData = GPy.models.GPRegression(x2, yc2, kernel=kernel)

    ''' Optimization '''
    m_tobit.optimize(optimizer='lbfgs', max_iters=500, messages=True)
    m_normal.optimize(optimizer='lbfgs', max_iters=500, messages=True)
    m_normalWithoutCensoredData.optimize(optimizer='lbfgs', max_iters=500, messages=True)

    ''' Plots '''
    plotModels(m_tobit, m_normal, m_normalWithoutCensoredData)

if __name__ == "__main__":
    artificialExample()
