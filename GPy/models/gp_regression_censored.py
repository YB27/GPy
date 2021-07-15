import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern

class GPRegressionCensored(GP):

    def __init__(self, X, Y, lowerThreshold, upperThreshold, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Tobit(lowerThreshold=lowerThreshold, upperThreshold=upperThreshold, variance=noise_var)

        super(GPRegressionCensored, self).__init__(X, Y, kernel, likelihood, name='GP regression censored', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        return GPRegressionCensored(gp.X, gp.Y, gp.likelihood.l, gp.likelihood.u, gp.kern, gp.Y_metadata, gp.normalizer, gp.likelihood.variance.values,
                            gp.mean_function)

    def to_dict(self, save_data=True):
        model_dict = super(GPRegressionCensored,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPRegressionCensored"
        return model_dict

    @staticmethod
    def _from_dict(input_dict, data=None):
        import GPy
        input_dict["class"] = "GPy.core.GP"
        m = GPy.core.GP.from_dict(input_dict, data)
        return GPRegressionCensored.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)
