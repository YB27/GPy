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

    def __init__(self, lowerThreshold, upperThreshold, variance=1., gp_link=None):
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
            self.l.fix()
            self.link_parameter(self.l)
        else:
            self.l = None
        if(upperThreshold is not None):
            self.u = Param('upperThreshold', upperThreshold)
            self.u.fix()
            self.link_parameter(self.u)
        else:
            self.u = None

        print("l : ")
        print(self.l)

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
        if(self.l is not None):
            a = (self.l - mu)/s
            phi_a = stats.norm.pdf(a)
            Phi_a = stats.norm.cdf(a)
        else:
            phi_a = 0.
            Phi_a = 0.

        if(self.u is not None):
            b = (self.u - mu)/s
            phi_b = stats.norm.pdf(b)
            Phi_b = stats.norm.cdf(b)
        else:
            phi_b = 0.
            Phi_b = 1.

        Ey = mu*(Phi_b - Phi_a) + s*(phi_a - phi_b)
        if(self.l is not None):
            Ey += self.l*Phi_a
        if(self.u is not None):
            Ey += self.u*(1.-Phi_b)

        print("tobit::predictive_mean: {}".format(Ey))
        return Ey

    def predictive_variance(self, mu, variance, predictive_mean=None, Y_metadata=None):
        '''
               Formula not given in the paper and not implemented in the Matlab source code
               See the doc "Proof of formula in Gaussian Process Regression with
               Censored Data Using Expectation Propagation, Groot, Lucas
               for the derivation of the formula
        '''
        s = np.sqrt(self.variance + variance)
        if(self.l is not None):
            a = (self.l - mu) / s
            phi_a = stats.norm.pdf(a)
            Phi_a = stats.norm.cdf(a)
        else:
            a = 0.
            phi_a = 0.
            Phi_a = 0.

        if(self.u is not None):
            b = (self.u - mu) / s
            phi_b = stats.norm.pdf(b)
            Phi_b = stats.norm.cdf(b)
        else:
            b = 0.
            phi_b = 0.
            Phi_b = 1.

        #Vy = mu*(Phi_b - Phi_a) + s*(a*phi_a - Phi_a - b*phi_b + Phi_b) - predictive_mean**2
        Vy = (Phi_b -Phi_a)*mu**2 + 2.*s*mu*(phi_a - phi_b) + (a*phi_a - Phi_a - b*phi_b + Phi_b)*s**2 - predictive_mean**2
        if(self.l is not None):
            Vy += Phi_a * self.l ** 2
        if(self.u is not None):
            Vy += (1. - Phi_b)*self.u ** 2

        print("tobit::predictive_variance : {}".format(Vy))
        return Vy

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        pred_mean = self.predictive_mean(mu, var)
        pred_var = self.predictive_variance(mu, var, pred_mean)
        return [stats.norm.ppf(q / 100.) * np.sqrt(pred_var) + pred_mean for q in quantiles]

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
        sigma2_i = 1./tau_i
        mu_i = v_i*sigma2_i
        sum_var = self.variance + sigma2_i

        if(self.l is None):
            l = np.NINF
        else:
            l = self.l
        if(self.u is None):
            u = np.PINF
        else:
            u = self.u

        if(data_i <= l):
            print("data_i < l")
            sum_var_root = np.sqrt(sum_var)
            z_il = (mu_i - l) / sum_var_root
            ''' Probit moments '''
            mp_0 = stats.norm.cdf(z_il)
            phi_Phi_zil = self.ratio_phiPhi(z_il)
            mp_1 = mu_i + (sigma2_i*phi_Phi_zil)/sum_var_root
            mp_2 = sigma2_i - ((sigma2_i**2)*phi_Phi_zil)/sum_var*(z_il + phi_Phi_zil)

            ''' Tobit moments '''
            Z_hat = 1. - mp_0
            phi_Phi_zil_minus = self.ratio_phiPhi(-z_il)
            mu_hat = mu_i - (sigma2_i*phi_Phi_zil_minus)/sum_var_root
            sigma2_hat = ((sigma2_i + mu_i**2) - mp_0*(mp_2 + mp_1**2))/Z_hat - mu_hat**2
        elif(data_i >= u):
            print("data_i > u")
            z_il = (mu_i - u) / sum_var_root
            phi_Phi_zil = self.ratio_phiPhi(z_il)
            Z_hat = stats.norm.cdf(z_il)
            mu_hat = mu_i + (sigma2_i*phi_Phi_zil)/sum_var_root
            sigma2_hat = sigma2_i - (phi_Phi_zil*sigma2_i**2)*(z_il + phi_Phi_zil)/sum_var
        else:
            ''' Gaussian moments '''
            Z_hat = 1. / np.sqrt(2. * np.pi * sum_var) * np.exp(-.5 * (data_i - mu_i) ** 2. / sum_var)
            sigma2_hat = sigma2_i - sigma2_i**2 / sum_var #1. / (1. / self.variance + tau_i)
            mu_hat = mu_i + sigma2_i*((data_i - mu_i)/ sum_var) #sigma2_hat * (data_i / self.variance + v_i)

        #print("Tobit::moments_match_ep (data_i, tau_i, v_i, Z_hat, mu_hat, sigma2_hat) : {}, {}, {}, {}, {}, {}".format(data_i, tau_i, v_i, Z_hat, mu_hat, sigma2_hat))
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

    def logpdf_link_lowerCensored(self, link_f):
        return stats.norm.logcdf((self.l - link_f)/np.sqrt(self.variance))

    def logpdf_link_upperCensored(self, link_f):
        return stats.norm.logcdf((link_f - self.u)/np.sqrt(self.variance))

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
        res = np.empty((y.shape[0],1))
        inv_std = 1. / np.sqrt(self.variance)

        assert( ('lowerCensored' in Y_metadata.keys() or 'upperCensored' in Y_metadata.keys())
               and 'gaussianIndexes' in Y_metadata.keys())

        ''' TODO : maybe store the indexes corresponding to the three cases once in Y_metadata '''
        #idxValTuples = enumerate(y)
        y_l_indexes = Y_metadata['lowerCensored'] #[idx for idx, val in idxValTuples if val == l]
        y_u_indexes = Y_metadata['upperCensored'] #[idx for idx_val in idxValTuples if val == u]
        y_indexes = Y_metadata['gaussianIndexes'] #[idx for idx, val in idxValTuples if val > u and val < l]

        res[y_l_indexes] = self.logpdf_link_lowerCensored(link_f[y_l_indexes]) # stats.norm.logcdf((self.l - link_f[y_l_indexes]) * inv_std)
        res[y_u_indexes] = self.logpdf_link_upperCensored(link_f[y_u_indexes]) #stats.norm.logcdf((link_f[y_u_indexes] - self.u) * inv_std)
        res[y_indexes] = -(1.0/(2*self.variance))*((y[y_indexes]-link_f[y_indexes])**2) - 0.5*np.log(self.variance) - 0.5*np.log(2.*np.pi)
        return res

    def dlogpdf_dlink_lowerCensored(self, link_f):
        zl = (link_f - self.l)/np.sqrt(self.variance)
        return -self.ratio_phiPhi(-zl)

    def dlogpdf_dlink_upperCensored(self, link_f):
        zu = (link_f - self.u)/np.sqrt(self.variance)
        return self.ratio_phiPhi(zu)

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
        grad = np.empty((y.shape[0], link_f.shape[1]))

        assert (('lowerCensored' in Y_metadata.keys() or 'upperCensored' in Y_metadata.keys())
                and 'gaussianIndexes' in Y_metadata.keys())

        ''' TODO : maybe store the indexes corresponding to the three cases once in Y_metadata '''
        # idxValTuples = enumerate(y)
        y_l_indexes = Y_metadata['lowerCensored']  # [idx for idx, val in idxValTuples if val == l]
        y_u_indexes = Y_metadata['upperCensored']  # [idx for idx_val in idxValTuples if val == u]
        y_indexes = Y_metadata['gaussianIndexes']  # [idx for idx, val in idxValTuples if val > u and val < l]

        inv_std = 1./np.sqrt(self.variance)
        grad[y_l_indexes] = self.dlogpdf_dlink_lowerCensored(link_f[y_l_indexes]) # -self.ratio_phiPhi(-zl)
        grad[y_u_indexes] = self.dlogpdf_dlink_upperCensored(link_f[y_u_indexes]) #self.ratio_phiPhi(zu)
        grad[y_indexes] = (y[y_indexes] - link_f[y_indexes])*inv_std

        return inv_std*grad

    def d2logpdf_dlink2_lowerCensored(self, link_f):
        sigma = np.sqrt(self.variance)
        z_l = (link_f - self.l) / sigma
        phi_l = stats.norm.pdf(z_l)
        Phi_l = stats.norm.cdf(z_l)
        one_minus_Phi_l = np.ones((link_f.shape[0], 1)) - Phi_l
        return link_f*phi_l*(one_minus_Phi_l - phi_l)/(self.variance*(one_minus_Phi_l**2))

    def d2logpdf_dlink2_upperCensored(self, link_f):
        sigma = np.sqrt(self.variance)
        z_u = (link_f - self.u) / sigma
        phi_u = stats.norm.pdf(z_u)
        Phi_u = stats.norm.cdf(z_u)
        return -link_f * phi_u * (phi_u + Phi_u) / (self.variance * Phi_u ** 2)

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link_f, w.r.t link_f.
        i.e. second derivative logpdf at y given link(f_i) link(f_j)  w.r.t link(f_i) and link(f_j)

        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{2}f} = -\\frac{1}{\\sigma^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points link(f))
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """

        assert (('lowerCensored' in Y_metadata.keys() or 'upperCensored' in Y_metadata.keys())
                and 'gaussianIndexes' in Y_metadata.keys())
        hess = np.empty((y.shape[0], 1))

        ''' TODO : maybe store the indexes corresponding to the three cases once in Y_metadata '''
        # idxValTuples = enumerate(y)
        y_indexes = Y_metadata['gaussianIndexes']  # [idx for idx, val in idxValTuples if val > u and val < l]

        if(self.l is not None and 'lowerCensored' in Y_metadata.keys()):
            y_l_indexes = Y_metadata['lowerCensored']  # [idx for idx, val in idxValTuples if val == l]
            hess[y_l_indexes] = self.d2logpdf_dlink2_lowerCensored(link_f[y_l_indexes]) #link_f[y_indexes]*phi_l*(one_minus_Phi_l - phi_l)/(self.variance*(one_minus_Phi_l**2))
        if(self.u is not None and 'upperThreshold' in Y_metadata.keys()):
            y_u_indexes = Y_metadata['upperCensored']  # [idx for idx_val in idxValTuples if val == u]
            hess[y_u_indexes] = self.d2logpdf_dlink2_upperCensored(link_f[y_u_indexes]) #-link_f[y_u_indexes]*phi_u*(phi_u + Phi_u)/(self.variance*Phi_u**2)

        hess[y_indexes] = -(1.0/self.variance)*np.ones((y_indexes.shape[0], 1))
        return hess

    def dlogpdf_link_dvar_lowerCensored(self, link_f):
        zl = (link_f - self.l) / np.sqrt(self.variance)
        return (0.5/self.variance) * zl * self.ratio_phiPhi(-zl)

    def dlogpdf_link_dvar_upperCensored(self, link_f):
        zu = (link_f - self.u) / np.sqrt(self.variance)
        return (0.5/self.variance) * zu * self.ratio_phiPhi(zu)

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
        assert (('lowerCensored' in Y_metadata.keys() or 'upperCensored' in Y_metadata.keys())
                and 'gaussianIndexes' in Y_metadata.keys())

        grad = np.empty((link_f.shape[0],1))

        ''' TODO : maybe store the indexes corresponding to the three cases once in Y_metadata '''
        # idxValTuples = enumerate(y)
        y_indexes = Y_metadata['gaussianIndexes']  # [idx for idx, val in idxValTuples if val > u and val < l]

        if(self.l is not None and 'lowerCensored' in Y_metadata.keys()):
            y_l_indexes = Y_metadata['lowerCensored']  # [idx for idx, val in idxValTuples if val == l]
            grad[y_l_indexes] = self.dlogpdf_link_dvar_lowerCensored(link_f[y_l_indexes])

        if(self.u is not None and 'upperCensored' in Y_metadata.keys()):
            y_u_indexes = Y_metadata['upperCensored']  # [idx for idx_val in idxValTuples if val == u]
            grad[y_u_indexes] = self.dlogpdf_link_dvar_upperCensored(link_f[y_u_indexes])

        grad[y_indexes] = (0.5/self.variance)*((y[y_indexes] - link_f[y_indexes])**2/self.variance - 1.)
        return grad

    def variational_expectations_censored(self, X_reshaped, X_shape, gh_w, shape, logp, dlogp_dx, d2logp_dx2):
        # average over the grid to get derivatives of the Gaussian's parameters
        # division by pi comes from fact that for each quadrature we need to scale by 1/sqrt(pi)
        F = np.dot(logp, gh_w) / np.sqrt(np.pi)
        dF_dm = np.dot(dlogp_dx, gh_w) / np.sqrt(np.pi)
        dF_dv = np.dot(d2logp_dx2, gh_w) / np.sqrt(np.pi)
        dF_dv /= 2.

        if np.any(np.isnan(dF_dv)) or np.any(np.isinf(dF_dv)):
            stop
        if np.any(np.isnan(dF_dm)) or np.any(np.isinf(dF_dm)):
            stop

        if self.size:
            dF_dtheta = self.dlogpdf_link_dvar_lowerCensored(X_reshaped).reshape(X_shape)  # Ntheta x (orig size) x N_{quad_points}
            dF_dtheta = np.dot(dF_dtheta, gh_w) / np.sqrt(np.pi)
            dF_dtheta = dF_dtheta.reshape(self.size, shape[0], shape[1])
        else:
            dF_dtheta = None  # Not yet implemented

        return F, dF_dmu, dF_dv, dF_dtheta

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        keys = Y_metadata.keys()
        assert('gaussianIndexes' in keys and ('lowerCensored' in Y_metadata or 'upperCensored' in Y_metadata))
        n = Y.shape[0]
        F = np.empty((n,1))
        dF_dmu = np.empty((n,1))
        dF_dv = np.empty((n,1))
        dF_dtheta = np.empty((self.size, n, m.shape[1]))

        ''' For non-censored data -> gaussian case '''
        y_indexes = Y_metadata['gaussianIndexes']  # [idx for idx, val in idxValTuples if val > u and val < l]
        lik_var = float(self.variance)
        Y_indexes = Y[y_indexes]
        m_indexes = m[y_indexes]
        v_indexes = v[y_indexes]

        F[y_indexes] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(lik_var) - 0.5 * (
                    np.square(Y_indexes) + np.square(m_indexes) + v_indexes - 2 * m_indexes * Y_indexes) / lik_var
        dF_dmu[y_indexes] = (Y_indexes - m_indexes) / lik_var
        dF_dv[y_indexes] = np.ones_like(v_indexes) * (-0.5 / lik_var)
        dF_dtheta[:,y_indexes,:] = -0.5 / lik_var + 0.5 * (np.square(Y_indexes) + np.square(m_indexes) + v_indexes - 2 * m_indexes * Y_indexes) / (lik_var ** 2)

        ''' Here, no analytical expressions so use Gaussian quadrature (as in likelihoods.variational_expectations) '''
        if gh_points is None:
            gh_x, gh_w = self._gh_points()
        else:
            gh_x, gh_w = gh_points
        if(self.l is not None and 'lowerCensored' in keys):
            y_l_indexes = Y_metadata['lowerCensored']  # [idx for idx, val in idxValTuples if val == l]
            shape = m[y_l_indexes].shape
            m_l, v_l = m[y_l_indexes].reshape((shape[0],1)), v[y_l_indexes].reshape((shape[0],1))

            # make a grid of points
            X = gh_x[None, :] * np.sqrt(2. * v_l[:, None]) + m_l[:, None]
            n = X.shape[0] * X.shape[1] * X.shape[2]
            X_reshaped = X.reshape((n,1))

            # evaluate the likelihood for the grid. First ax indexes the data (and mu, var) and the second indexes the grid.
            # broadcast needs to be handled carefully.
            logp = self.logpdf_link_lowerCensored(X_reshaped).reshape(X.shape[0], X.shape[1], X.shape[2])
            dlogp_dx = self.dlogpdf_dlink_lowerCensored(X_reshaped).reshape(X.shape[0], X.shape[1], X.shape[2])
            d2logp_dx2 = self.d2logpdf_dlink2_lowerCensored(X_reshaped).reshape(X.shape[0], X.shape[1], X.shape[2])

            F[y_l_indexes], dF_dmu[y_l_indexes], dF_dv[y_l_indexes], dF_dtheta[:,y_l_indexes,:] = self.variational_expectations_censored(X_reshaped, X.shape, gh_w, shape, logp, dlogp_dx, d2logp_dx2)

        if(self.u is not None and 'upperThreshold' in keys):
            y_u_indexes = Y_metadata['upperCensored']  # [idx for idx_val in idxValTuples if val == u]
            shape = m[y_u_indexes].shape
            m_u, v_u = m[y_u_indexes].reshape((shape[0],1)), v[y_u_indexes].reshape((shape[0],1))

            # make a grid of points
            X = gh_x[None, :] * np.sqrt(2. * v_u[:, None]) + m_u[:, None]

            # evaluate the likelihood for the grid. First ax indexes the data (and mu, var) and the second indexes the grid.
            # broadcast needs to be handled carefully.
            logp = self.logpdf_link_upperCensored(X)
            dlogp_dx = self.dlogpdf_dlink_upperCensored(X)
            d2logp_dx2 = self.d2logpdf_dlink2_upperCensored(X)

            F[y_u_indexes], dF_dmu[y_u_indexes], dF_dv[y_u_indexes], dF_dtheta[:,y_u_indexes,:] = variational_expectations_censored(X, gh_w, shape, logp, dlogp_dx, d2logp_dx2)

        return F, dF_dmu, dF_dv, dF_dtheta

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
