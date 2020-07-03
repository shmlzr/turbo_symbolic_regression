"""
Elastic Net adaption for Symbolic Regression

2019, M.Schmelzer
"""
import numpy as np
from sklearn.base import RegressorMixin
import sklearn.preprocessing as preprocessing
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.exceptions import ConvergenceWarning
import warnings as warn
from turbosparsereg.methods.symbolic_regression import SymbolicRegression
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
from sklearn.linear_model.base import LinearModel


class TSRElasticNet(LinearModel, RegressorMixin, SymbolicRegression):
    """ Model discovery engine using sklearn.linear_model.ElasticNet:
        "Linear regression with combined L1 and L2 priors as regularizer."

    The main concept comes from the FFX algorithm by Trent McConaghy
        (see: FFX: Fast, Scalable, Deterministic Symbolic Regression Technology. In: 2 GENETIC PROGRAMMING THEORY AND
        PRACTICE VI. 2011)


    Parameters
    ----------
    n_alphas : int, optional
        Length of vector with alphas (penalty term multiplier)

    max_iter : int, optional
        Maximal number of iterations for coordinate descent

    fit_intercept : bool, optional
        Fitting the intercept.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    print_warnings: boolean, optional
        When set to ``True`` warnings are printed out.


    Attributes
    ----------
    model_structures_: array, shape (n_model, n_active_features)
        Vector with booleans indicating active/unactive features of the library

    alphas_ : array, shape (n_alphas,)
        Grid of alpha values for elasticnet parameter search
    """
    def __init__(self, n_alphas=100, max_iter=1000, standardization=True, l1_ratios=[.01, .1, .2, .5, .7, .9, .95, .99, 1.0],
                 ridge_alphas=[0, 1.0, 0.1, 0.01, 0.001], fit_intercept=False, warm_start=True,
                 print_warnings=False):
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.standardization = standardization
        self.l1_ratios = l1_ratios
        self.ridge_alphas = ridge_alphas
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.print_warnings = print_warnings
        self.model_structures_ = None

    def _create_enet(self, x, y):
        """ Creates a grid of alpha values for elasticnet parameter search using _alpha_grid()

        :param x: Training data (n_samples, n_features)
        :param y: Target values (n_samples,)
        :return: alphas_ (n_alphas,)
        """
        self.alphas_ = _alpha_grid(x, y, n_alphas=self.n_alphas)
        return self

    def _prepare(self, x_train):
        """ Scaling training data using StandardScaler
        :param x: Training data (n_samples, n_features)
        :param y: Target values (n_samples,)
        :return: x_scaled (n_samples, n_features)
        """
        if self.standardization:
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_scaled = scaler.transform(x_train)
        else:
            x_scaled = x_train
        return x_scaled

    def fit(self, x, y):
        """ Identify model structures by fitting model with coordinate descent.

        :param x: Training data (n_samples, n_features)
        :param y: Target values (n_samples,)
        :return: model_structures_ (n_model, n_active_features)
        """
        # Perform standardization of the input data x
        x = self._prepare(x)

        # Define elastic net
        self._create_enet(x, y)

        if not self.print_warnings:
            # Warnings can safely be ignored as this step is only meant for model discovery not calibration
            warn.filterwarnings("ignore", category=ConvergenceWarning)

        print("\nDoing model discovery using elastic net regression...")
        # For each (l1_ratio, alpha) combination the optimization problem is solved
        model_structures = []
        for i, alpha in enumerate(self.alphas_):
            print("i =", i+1, "of", len(self.alphas_))

            for j, l1_ratio in enumerate(self.l1_ratios):
                eln = ElasticNet(l1_ratio=l1_ratio, alpha=alpha, max_iter=self.max_iter,
                                 fit_intercept=self.fit_intercept, warm_start=self.warm_start)
                eln.fit(x, y)

                # Ignore over-regularized solutions
    #            if not all(eln.coef_ == 0.0):
    #                model_structures.append(abs(eln.coef_) > 0.0)
                model_structures.append(eln.coef_)


        # Only retain unique model structures
        self.model_structures_ = np.unique(model_structures, axis=0)
        #self.model_structures_ = model_structures

        return self
