"""
Symbolic Regression Methods

2019, M.Schmelzer
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.utils.extmath import safe_sparse_dot


class SymbolicRegression:
    """ A base class for symbolic regression methods.
    """

    @staticmethod
    def build_model(coefs, candidate_library):
        """ Writes mathematical model as string.

        :param coefs: Coefficient vector (n_features,)
        :param candidate_library: Symbolic library of candidate functions (n_features,)
        :return: model_string
        """
        i_nonzero = np.nonzero(coefs)[0]
        model_string = ''
        for i in i_nonzero:
            model_string = model_string + ' + ' + str(coefs[i]) + '*' + candidate_library[i]
        return model_string

    def model_inference(self, x_train, x_test, y_train, y_test, candidate_library):
        """ Performs inference of model coefficients using Ridge regression.

        :param x_train: Training data (xi*n_samples, n_features)
        :param x_test: Testing data ((1-xi)*n_samples, n_features)
        :param y_train: Target values for training (xi*n_samples,)
        :param y_test: Target values for testing ((1-xi)*n_samples,)
        :param candidate_library: Symbolic library of candidate functions (n_features,)
        :return: models: pandas dataframe
        """
        print("\nModel inference using ridge regression")

        num_candidates = self.model_structures_.shape[-1]
        models = pd.DataFrame()

        for ridge_alpha in self.ridge_alphas:
            print("Ridge alpha =", ridge_alpha, "\n")

            for model_structure in self.model_structures_:
                i_nonzero = np.where(model_structure)[0]
                # Perform regression (OLS or RIDGE) on the active features only (nonzero coef)
                tmp_ = np.zeros(num_candidates)

                if ridge_alpha == 0.0:
                    model = LinearRegression(fit_intercept=False, normalize=False)

                else:
                    model = Ridge(alpha=ridge_alpha, fit_intercept=False, normalize=False)

                model.fit(x_train[:, i_nonzero], y_train)
                tmp_[i_nonzero] = model.coef_

                models = models.append({
                    'coef_': tmp_,
                    'str_': self.build_model(np.round(tmp_, 5), candidate_library),
                    'mse_': np.average((y_test - safe_sparse_dot(x_test, tmp_)) ** 2.0),
                    'complexity_': np.count_nonzero(tmp_),
                    'l1_norm_': np.linalg.norm(tmp_, ord=1),
                    'model': model,
                    'model_structure_':model_structure,
                    'ridge_alpha': ridge_alpha
                }, ignore_index=True)

        return models
