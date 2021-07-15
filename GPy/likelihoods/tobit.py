import numpy as np
from scipy import stats, special
from . import link_functions
from .likelihood import Likelihood
from ..core.parameterization import Param
from paramz.transformations import Logexp
from scipy import stats

class Tobit(Likelihood):
    """
    Tobit likelihood (Type I) with right and left censoring with lower and upper threshold l and u.

    .. math::
        \\p(\\mathbf{y}|\\mathbf{f}) = \\prod_{i=1}^{n} p(y_{i}|f_{i}) = \\prod_{y_i = l} \\left(1 - \\Phi(\\frac{f_i - l}{\\sigma})\\right)
        \\prod_{l \\leq y_{i} \\leq u} \\left(\\frac{1}{\\sigma} \\phi\\left(\\frac{y_{i} - f_{i}}{\\sigma}\\right)\\right)
        \\prod_{y_{i} = u} \\Phi\\left(\\frac{f_{i} - u}{\\sigma}\\right)

    """

    def __init__(self, lowerThreshold, upperThreshold, variance, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        if not isinstance(gp_link, link_functions.Identity):
            print("Warning, Exact inference is not implemented for non-identity link functions,\
            if you are not already, ensure Laplace inference_method is used")

        super(Tobit, self).__init__(gp_link, name="Tobit")

        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)
        if(lowerThreshold is not None):
            self.l = Param('lowerThreshold', lowerThreshold)
            self.link_parameter(self.l)
        else:
            self.l = None
        if(upperThreshold is not None):
            self.u = Param('upperThreshold', upperThreshold)
            self.link_parameter(self.u)
        else:
            self.u = None


    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Tobit, self)._save_to_input_dict()
        input_dict["class"] = "GPy.likelihoods.Tobit"
        input_dict["variance"] = self.variance.values.tolist()
        if(self.l is not None):
            input_dict["lowerThreshold"] = self.l.values.tolist()
        else:
            input_dict["lowerThreshold"] = None
        if(self.u is not None):
            input_dict["upperThreshold"] = self.u.values.tolist()
        else:
            input_dict["upperThreshold"] = None
        return input_dict

    def gaussian_variance(self):
        return self.variance

    def predictive_mean(self, mu, variance, Y_metadata=None):
        '''
        Formula not given in the paper but present in the Matlab source code
        See the doc "Proof of formula in Gaussian Process Regression with
        Censored Data Using Expectation Propagation, Groot, Lucas
        for the derivation of the formula
        '''
        s = np.sqrt(self.variance + variance)
        a = (self.l - mu)/s
        b = (self.u - mu)/s
        phi_a = stats.norm.pdf(a)
        phi_b = stats.norm.pdf(b)
        Phi_a = stats.norm.cdf(a)
        Phi_b = stats.norm.cdf(b)

        Ey = mu*(Phi_b - Phi_a) + s*(phi_a - phi_b)
        if(self.l is not None):
            Ey += self.l*Phi_a
        if(self.u is not None):
            Ey += self.u*(1-Phi_b)
        return Ey

    def predictive_variance(self, mu, variance, predictive_mean=None, Y_metadata=None):
        '''
               Formula not given in the paper and not implemented in the Matlab source code
               See the doc "Proof of formula in Gaussian Process Regression with
               Censored Data Using Expectation Propagation, Groot, Lucas
               for the derivation of the formula
        '''

        s = np.sqrt(self.variance + variance)
        a = (self.l - mu) / s
        b = (self.u - mu) / s
        phi_a = stats.norm.pdf(a)
        phi_b = stats.norm.pdf(b)
        Phi_a = stats.norm.cdf(a)
        Phi_b = stats.norm.cdf(b)

        Vy = mu*(Phi_b - Phi_a) + s*(Phi_b*b**2 - Phi_a*a**2 + 2.*(phi_b - phi_a)) - predictive_mean**2
        if(self.l is not None):
            Vy +=  Phi_a * l ** 2
        if(self.u is not None):
            Vy += (1 - Phi_b)*u ** 2

        return Vy

    def update_gradients(self, grad):
        self.variance.gradient = grad

    def ep_gradients(self, Y, cav_tau, cav_v, dL_dKdiag, Y_metadata=None, quad_mode='gk', boost_grad=1.):
        return self.exact_inference_gradients(dL_dKdiag)

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata=None):
        return dL_dKdiag.sum()

    def moments_match_ep(self, data_i, tau_i, v_i, Y_metadata_i=None):
        """
        Moments match of the marginal approximation in EP algorithm
        Based on the matlab implementation of "Gaussian Process Regression with
        Censored Data Using Expectation Propagation, P. Groot, P. Lucas"

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        ''' Variance and mean of the cavity distribution'''
        sigma_i = 1./tau_i
        sigma2_i = sigma_i**2
        mu_i = v_i*sigma_i
        sum_var = self.variance + sigma2_i

        if(data_i == self.l):
            sum_var_root = np.sqrt(sum_var)
            z_il = (mu_i - self.l) / sum_var_root
            ''' Probit moments '''
            mp_0 = stats.norm.cdf(z_il)
            phi_Phi_zil = self.ratio_phiPhi(z_il)
            mp_1 = mu_i + (sigma2_i*phi_Phi_zil)/sum_var_root
            mp_2 = sigma2_i - ((sigma2_i**2)*phi_Phi_zil)/((z_il + phi_Phi_zil)*sum_var)

            ''' Tobit moments '''
            Z_hat = 1. - mp_0
            phi_Phi_zil_minus = self.ratio_phiPhi(-z_il)
            mu_hat = mu_i - (sigma2_i*phi_Phi_zil_minus)/sum_var_root
            sigma2_hat = ((sigma2_i + mu_i**2) - mp_0*(mp_2 + mp_1**2))/Z_hat - mu_hat**2
        elif(data_i == self.u):
            z_il = (mu_i - self.u) / sum_var_root
            phi_Phi_zil = self.ratio_phiPhi(z_il)
            Z_hat = stats.norm.cdf(z_il)
            mu_hat = mu_i + (sigma2_i*phi_Phi_zil)/sum_var_root
            sigma2_hat = sigma2_i - (phi_Phi_zil*sigma2_i**2)*(z_il + phi_Phi_zil)/sum_var
        else:
            ''' Gaussian moments '''
            Z_hat = 1. / np.sqrt(2. * np.pi * sum_var) * np.exp(-.5 * (data_i - mu_i) ** 2. / sum_var)
            sigma2_hat = 1. / (1. / self.variance + tau_i)
            mu_hat = sigma2_hat * (data_i / self.variance + v_i)

        return Z_hat, mu_hat, sigma2_hat

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        return np.exp(self.logpdf_link(link_f, y, Y_metadata))

    def logpdf_link(self, link_f, y, Y_metadata):
        """
        Log likelihood function given link(f)

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: log likelihood evaluated for this point
        :rtype: float
        """
        inv_std = 1. / np.sqrt(self.variance)

        assert('lowerCensored' in Y_metadata.keys() and 'upperCensored' in Y_metadata.keys()
               and 'gaussianIndexes' in Y_metadata.keys())

        ''' TODO : maybe store the indexes corresponding to the three cases once in Y_metadata '''
        #idxValTuples = enumerate(y)
        y_l_indexes = Y_metadata['lowerCensored'] #[idx for idx, val in idxValTuples if val == l]
        y_u_indexes = Y_metadata['upperCensored'] #[idx for idx_val in idxValTuples if val == u]
        y_indexes = Y_metadata['gaussianIndexes'] #[idx for idx, val in idxValTuples if val > u and val < l]

        lowerCensoringSum = np.sum(stats.norm.logcdf((self.l - link_f[y_l_indexes]) * inv_std))
        upperCensoringSum = np.sum(stats.norm.logcdf((link_f[y_u_indexes] - self.u) * inv_std))
        gaussianSum = np.exp(np.sum(np.log(stats.norm.pdf(y[y_indexes], link_f[y_indexes], np.sqrt(self.variance)))))
        return lowerCensoringSum + upperCensoringSum + gaussianSum

    ''' Formula taken from the supplementary material of "Gaussian Process Regression with
        Censored Data Using Expectation Propagation, P. Groot, P. Lucas" '''
    def dlogpdf_dlink(self, link_f, y, Y_metadata):
        """
        Gradient of the pdf at y, given link(f) w.r.t link(f)

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: gradient of log likelihood evaluated at points link(f)
        :rtype: Nx1 array
        """

        grad = np.empty((y.shape[0],))

        assert ('lowerCensored' in Y_metadata.keys() and 'upperCensored' in Y_metadata.keys()
                and 'gaussianIndexes' in Y_metadata.keys())

        ''' TODO : maybe store the indexes corresponding to the three cases once in Y_metadata '''
        # idxValTuples = enumerate(y)
        y_l_indexes = Y_metadata['lowerCensored']  # [idx for idx, val in idxValTuples if val == l]
        y_u_indexes = Y_metadata['upperCensored']  # [idx for idx_val in idxValTuples if val == u]
        y_indexes = Y_metadata['gaussianIndexes']  # [idx for idx, val in idxValTuples if val > u and val < l]

        inv_std = 1./np.sqrt(self.variance)
        zl = (link_f[y_l_indexes] - self.l)*inv_std
        zu = (link_f[y_u_indexes] - self.u)*inv_std
        grad[y_l_indexes] = -self.ratio_phiPhi(-zl)
        grad[y_u_indexes] = self.ratio_phiPhi(zu)
        grad[y_indexes] = (y[y_indexes] - link_f[y_indexes])*inv_std

        return inv_std*grad

    def dlogpdf_link_dvar(self, link_f, y, Y_metadata):
        """
        Gradient of the log-likelihood function at y given link(f), w.r.t variance parameter (noise_variance)

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: float
        """

        assert ('lowerCensored' in Y_metadata.keys() and 'upperCensored' in Y_metadata.keys()
                and 'gaussianIndexes' in Y_metadata.keys())

        ''' TODO : maybe store the indexes corresponding to the three cases once in Y_metadata '''
        # idxValTuples = enumerate(y)
        y_l_indexes = Y_metadata['lowerCensored']  # [idx for idx, val in idxValTuples if val == l]
        y_u_indexes = Y_metadata['upperCensored']  # [idx for idx_val in idxValTuples if val == u]
        y_indexes = Y_metadata['gaussianIndexes']  # [idx for idx, val in idxValTuples if val > u and val < l]

        zl = (link_f[y_l_indexes] - self.l)*inv_std
        zu = (link_f[y_u_indexes] - self.u)*inv_std
        dlik_dsigma_lowerCensoring = np.sum(zl*self.ratio_phiPhi(-zl))
        dlik_dsigma_upperCensoring = np.sum(zu*self.ratio_phiPhi(zu))
        dlik_dsigma_gaussian = np.sum((y[y_indexes] - link_f[y_indexes])**2/self.variance - 1)
        return (0.5/self.variance)*(dlik_dsigma_lowerCensoring - dlik_dsigma_upperCensoring + dlik_dsigma_gaussian)

    def ratio_phiPhi(self, z):
        '''
         Function to compute the ration phi/Phi where phi is the standart normal pdf and Phi the correspopnding cdf.
         Use an approximation for values near zero as both values converge to zero
        '''
        idxValTuples = enumerate(z)
        idx_low = [idx for idx, val in idxValTuples if val < -10]
        res = np.empty((z.shape[0],))
        if(len(idx_low) == 0):
            ''' Compute normally '''
            res = stats.norm.pdf(z)/stats.norm.cdf(z)
        else:
            ''' Use an asymptotic expansion '''
            z_low = z[idx_low]
            other_idx = [idx for idx, val in idxValTuples if val >= -10]
            z2 = z_low**2
            c = 1 - 1. / z2 * (1. - 3. / z2 * (1. - 5. / z2 * (1. - 7. / z2)))
            dc = 2. / z_low**3 - 12. / z_low**5 + 90. / z_low**7 - 840. / z_low**9
            res[idx_low] = -z_low - 1. / z_low + dc / c
            ''' For other values, compute normally '''
            res[other_idx] = stats.norm.pdf(z[other_idx]) / stats.norm.cdf(z[other_idx])

        return res