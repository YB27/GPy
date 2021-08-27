
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

def plotModels(xc, yc, xc2, yc2, m_tobit, m_normal, m_normalWithoutCensoredData):
    x_gt = np.linspace(0,1,200)
    y_gt = f(x_gt)

    fig, ax = plt.subplots()
    plt.title("Tobit GP model")
    plt.plot(x_gt, y_gt, linestyle='-', color="r", label="GT")
    plt.plot(xc, yc, linestyle='None', marker='+', markersize=10, color='k', label="Data")
    m_tobit.plot_f(fignum=0, ax=ax)
    plt.xlim([0, 1])

    fig, ax = plt.subplots()
    plt.title("Standart GP model")
    plt.plot(x_gt, y_gt,linestyle='-', color="r", label="GT")
    plt.plot(xc, yc, linestyle='None', marker='+', markersize=10, color='k', label="Data")
    m_normal.plot_f(fignum=1, ax=ax)
    plt.xlim([0,1])

    fig, ax = plt.subplots()
    plt.title("Standart ignoring censured data GP model")
    plt.plot(x_gt, y_gt, linestyle='-', color="r", label="GT")
    plt.plot(xc2, yc2, linestyle='None', marker='+', markersize=10, color='k', label="Data")
    m_normalWithoutCensoredData.plot_f(fignum=2, ax=ax)
    plt.xlim([0, 1])
    plt.show()

def artificialExample():
    ''' Generate Data '''
    np.random.seed(4)
    n = 30
    x = np.linspace(0,1,n)
    y = f(x) + np.random.normal(0, np.sqrt(0.1), x.shape[0])
    x = x.reshape((n,1))
    l =  -0.45 #-0.2265
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
    np.put(yc, lowerCensoredData_indexes, l)

    ''' Data without censored data'''
    yc2 = np.delete(yc, lowerCensoredData_indexes)
    x2 = np.delete(x,lowerCensoredData_indexes)
    yc = yc.reshape((n,1))
    yc2 = yc2.reshape(yc2.shape[0],1)
    x2 = x2.reshape(x2.shape[0],1)

    ''' GP models '''
    kernel_tobit = GPy.kern.RBF(input_dim=1, variance=5, lengthscale=0.1)
    kernel_normal = GPy.kern.RBF(input_dim=1, variance=5, lengthscale=0.1)
    kernel_normalCensored = GPy.kern.RBF(input_dim=1, variance=5, lengthscale=0.1)
    print("Create GP with tobit model ...")
    m_tobit = GPy.models.GPRegressionCensored(x, yc, lowerThreshold=l, upperThreshold=None, kernel=kernel_tobit, noise_var = 0.1, Y_metadata=y_metadata)
    m_tobit.likelihood.variance.fix()
    print("Create standart GP model ...")
    m_normal = GPy.models.GPRegression(x, yc, kernel=kernel_normal)
    m_normal.likelihood.variance.fix()
    print("Create standart GP model and ignoring censured data...")
    m_normalWithoutCensoredData = GPy.models.GPRegression(x2, yc2, kernel=kernel_normalCensored)
    m_normalWithoutCensoredData.likelihood.variance.fix()

    ''' Optimization '''
    print("Optimizer with tobit model ...")
    print("---> Model before opt : ")
    print(m_tobit[''])
    m_tobit.optimize(optimizer='lbfgs', max_iters=500, messages=True)
    print("---> Model after opt : ")
    print(m_tobit[''])
    print("Optimizer with standart model ...")
    print(m_normal[''])
    m_normal.optimize(optimizer='lbfgs', max_iters=500, messages=True)
    print("Optimizer with standart model and ignoring censured data...")
    m_normalWithoutCensoredData.optimize(optimizer='lbfgs', max_iters=500, messages=True)

    ''' Plots '''
    plotModels(x, yc, x2, yc2, m_tobit, m_normal, m_normalWithoutCensoredData)

if __name__ == "__main__":
    artificialExample()
