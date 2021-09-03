''' temporary : Use this to use the forked version of GPy'''
import sys
sys.path.insert(1, '/home/breux/GPy')

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import GPy
import climin
'''
Stochastic Variational Gaussian Processes regression with censored data example using artificial data as in 
"Gaussian Process Regression with Censored Data Using Expectation Propagation, P. Groot, P. Lucas

The example here is based on the GPy tutorial "Stochastic Variational Inference with Gaussian Processes"  
'''

def generateData():
    N = 500
    X = np.random.rand(N)[:, None]
    Y = np.sin(6 * X) + 0.1 * np.random.randn(N, 1)
    # Inducing points
    Z = np.random.rand(20, 1)

    return X, Y, Z

def generateCensoredData(Y, l, u):
    Y_metadata = {}
    Y_c = Y.copy()
    y_l_indexes, y_u_indexes = [], []
    if l is not None:
        y_l_indexes = [idx for idx, val in np.ndenumerate(Y.flatten()) if val <= l]
        #lowerCensoredData = np.zeros((n,), dtype='int64')
        #np.put(lowerCensoredData, y_l_indexes, 1)
        #np.put(gaussianData, y_l_indexes, 0)
        np.put(Y_c, y_l_indexes, l)
        Y_metadata["lowerCensored"] = np.array(y_l_indexes)
    if u is not None:
        y_u_indexes = [idx for idx, val in np.ndenumerate(Y.flatten()) if val >= u]
        #upperCensoredData = np.zeros((n,), dtype='int64')
        #np.put(upperCensoredData, y_u_indexes, 1)
        #np.put(gaussianData, y_u_indexes, 0)
        np.put(Y_c, y_u_indexes, u)
        Y_metadata["upperCensored"] = np.array(y_u_indexes)

    y_indexes = [idx for idx, val in np.ndenumerate(Y.flatten()) if idx not in y_l_indexes and idx not in y_u_indexes]
    Y_metadata["gaussianIndexes"] = np.array(y_indexes)

    return Y_c, Y_metadata

def optimizeAndPlotSVGP(m, X, Y, title):
    m.kern.white.variance = 1e-5
    m.kern.white.fix()

    def callback(i):
        print("Log-likelihood : {}".format(m.log_likelihood()))
        # Stop after 5000 iterations
        if i['n_iter'] > 5000:
            return True
        return False

    opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)
    info = opt.minimize_until(callback)
    print("Optimizer info : ")
    print(info)

    fig, ax = plt.subplots()
    ax.plot(X, Y, 'kx', alpha=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    m.plot_f(which_data_ycols=[0], plot_limits=(X.min(), X.max()), ax=ax)
    ax.set_xlim((X.min(), X.max()))
    fig.tight_layout()

def originalSVGP(X, Y, Z, batchsize):
    m = GPy.core.SVGP(X, Y, Z, GPy.kern.RBF(1) + GPy.kern.White(1), GPy.likelihoods.Gaussian(),
                      batchsize=batchsize)
    optimizeAndPlotSVGP(m, X, Y, "original SVGP")

def censoredSVGP(X, Y, Z, l, u, batchsize, Y_metadata):
    m = GPy.core.SVGPCensored(X, Y, Z, l, u, GPy.kern.RBF(1) + GPy.kern.White(1), batchsize=batchsize, Y_metadata=Y_metadata)
    optimizeAndPlotSVGP(m, X, Y, "lowerCensoredSVGP")

def lowerCensoredGP(X, Y, l, Y_metadata):
    m = GPy.models.GPRegressionCensored(X, Y, lowerThreshold=l, upperThreshold=None, kernel=GPy.kern.RBF(1) + GPy.kern.White(1), noise_var = 1, Y_metadata=Y_metadata)
    m.optimize(optimizer='lbfgs', max_iters=500, messages=True)
    fig, ax = plt.subplots()
    ax.plot(X, Y, 'kx', alpha=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("GP Censored")
    m.plot_f(which_data_ycols=[0], plot_limits=(X.min(), X.max()), ax=ax)
    ax.set_xlim((X.min(), X.max()))
    fig.tight_layout()

def example():
    l = None # -0.7
    u = 0.6
    X, Y, Z = generateData()
    Y_l, Y_metadata = generateCensoredData(Y, l=l, u=u)
    batchsize = 100
    print(" --> Original SVGP with censored data")
    #originalSVGP(X, Y_l, Z, batchsize)

    print("Lower censored SVGP")
    #Y_l, Y_metadata = generateCensoredData(Y, l=l, u=None)
    censoredSVGP(X, Y_l, Z, l, u, batchsize, Y_metadata)

    '''lowerCensoredData = np.zeros((Y.shape[0],), dtype='int64')
    lowerCensoredData_indexes = [idx for idx, val in np.ndenumerate(Y) if val < l]
    np.put(lowerCensoredData, lowerCensoredData_indexes, 1)
    gaussianData = np.zeros((Y.shape[0],), dtype='int64')
    gaussianData_indexes = [idx for idx, val in np.ndenumerate(Y) if val >= l]
    np.put(gaussianData, gaussianData_indexes, 1)
    Y_metadata_GP = {"lowerCensored": lowerCensoredData.reshape((Y.shape[0], 1)), "gaussianIndexes": gaussianData.reshape((Y.shape[0], 1))}
    lowerCensoredGP(X, Y_l, l, Y_metadata_GP)'''

    plt.show()

if __name__ == "__main__":
    example()