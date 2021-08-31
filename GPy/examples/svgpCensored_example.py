''' temporary : Use this to use the forked version of GPy'''
import sys
sys.path.insert(1, '/home/breux/GPy')

import numpy as np
import matplotlib.pyplot as plt
import GPy
import climin
'''
Stochastic Variational Gaussian Processes regression with censored data example using artificial data as in 
"Gaussian Process Regression with Censored Data Using Expectation Propagation, P. Groot, P. Lucas

The example here is based on the GPy tutorial "Stochastic Variational Inference with Gaussian Processes"  
'''

def generateData():
    N = 5000
    X = np.random.rand(N)[:, None]
    Y = np.sin(6 * X) + 0.1 * np.random.randn(N, 1)
    # Inducing points
    Z = np.random.rand(20, 1)

    return X, Y, Z

def generateCensoredData(Y, l, u):
    Y_metadata = {}
    Y_c = Y.copy()
    n = Y.shape[0]
    y_l_index, y_u_indexes = [],[]
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
    m.plot(which_data_ycols=[0], plot_limits=(X.min(), X.max()), ax=ax)
    ax.set_xlim((X.min(), X.max()))
    fig.tight_layout()

def originalSVGP(X, Y, Z, batchsize):
    m = GPy.core.SVGP(X, Y, Z, GPy.kern.RBF(1) + GPy.kern.White(1), GPy.likelihoods.Gaussian(),
                      batchsize=batchsize)
    optimizeAndPlotSVGP(m, X, Y, "original SVGP")

def lowerCensoredSVGP(X, Y, Z, l, batchsize, Y_metadata):
    m = GPy.core.SVGPCensored(X, Y, Z, l, None, GPy.kern.RBF(1) + GPy.kern.White(1), batchsize=batchsize, Y_metadata=Y_metadata)
    optimizeAndPlotSVGP(m, X, Y, "lowerCensoredSVGP")

def example():
    X, Y, Z = generateData()
    batchsize = 10
    print(" --> Original SVGP")
    #originalSVGP(X, Y, Z, batchsize)

    print("Lower censored SVGP")
    l = -0.6
    Y_l, Y_metadata = generateCensoredData(Y, l=l, u=None)
    lowerCensoredSVGP(X, Y_l, Z, 0.6, batchsize, Y_metadata)

    plt.show()

if __name__ == "__main__":
    example()