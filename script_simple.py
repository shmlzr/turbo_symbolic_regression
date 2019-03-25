"""
====================================================================================
Script to perform symbolic regression on a simple problem
====================================================================================
"""
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import turbosparsereg.dataprocessing.dataprocessor as dataproc
from turbosparsereg.util import util
from turbosparsereg.methods.sr_elasticnet import TSRElasticNet


def default_parameters():
    """ Defines default parameters for symbolic regression.
    """
    # Definition of raw features and mathematical operations
    operations = {'var': ['const', 'z'],
                  'exp': ['**-3.0', '**-2.0', '**-1.0', '**-0.5', '**0.5', '**1.0', '**2.0', '**3.0'],
                  'fun': ['abs', 'np.log10', 'np.exp', 'np.sin', 'np.cos']}

    # Which raw inputs should be used for the library
    active = {'var': '1100000', 'exp': '00000110', 'fun': '00010'}

    # Discrete structure of elastic net (at which value pairs l1/alpha should the optimization problem be solved)
    n_alphas = 100
    l1_ratios = [.01, .1, .2, .5, .7, .9, .95, .99, 1.0]

    # Set of ridge parameters for post-selection model inference
    ridge_alphas = [1e-5]

    return operations, active, n_alphas, l1_ratios, ridge_alphas


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Define parametric setup for data processing and symbolic regression
    # ------------------------------------------------------------------------------------------------------------------
    operations, active, n_alphas, l1_ratios, ridge_alphas = default_parameters()

    # # ----------------------------------------------------------------------------------------------------------------
    # # Create target data (toy problem)
    # # ----------------------------------------------------------------------------------------------------------------
    # raw input features
    z = np.linspace(-10, 0, 101)
    const = np.ones(101)

    # hidden process to be discovered
    hidden_model_str = '0.1*z**2.0 + z**2.0*np.sin(z) + 0.123*z'
    f_target = eval(hidden_model_str)

    # # ----------------------------------------------------------------------------------------------------------------
    # # Create library of candidate functions
    # # ----------------------------------------------------------------------------------------------------------------
    # lib = create_library(case, data_i, active, operations, do_create_library)
    buildlib = dataproc.BuildLibrary(operations)
    var_dict = {'const': const, 'z': z}
    B = buildlib.build_B_library(var_dict, active)
    B_data = np.stack([eval(B[i]) for i in range(len(B))]).T

    # # ----------------------------------------------------------------------------------------------------------------
    # # Perform sparse regression using ElasticNet
    # # ----------------------------------------------------------------------------------------------------------------
    ##
    # # Define generic variable names and subsample
    X = B_data
    y = f_target

    # Train/test split of the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Perform discovery of model structures (Which features are active?)
    tsr = TSRElasticNet(n_alphas=n_alphas, l1_ratios=l1_ratios, ridge_alphas=ridge_alphas).fit(x_train, y_train)

    # Perform model inference/calibration (Compute values of active coefficients)
    models = tsr.model_inference(x_train, x_test, y_train, y_test, B)

    # # ----------------------------------------------------------------------------------------------------------------
    # # Plot results
    # # ----------------------------------------------------------------------------------------------------------------
    ##
    # Plotting setup
    plt.close('all')
    util.init_plotting()

    # Extract models of a specific tuning parameter ridge_alpha
    query_inds = models.query("ridge_alpha==1e-5").index
    sorted_inds_ = models['mse_'].argsort()
    best_ind_ = sorted_inds_[0]

    # plot model matrix vs mean squared error
    plt.rcParams['figure.figsize'] = [3.2, 2.4]
    util.plot_model_and_mse(models, B, query_inds)
    plt.savefig('data/fig/SIMPLE_model_vs_mse.png', bbox_inches='tight')

    # Write out results in model_str.txt
    util.write_models_to_txt('data/models/SIMPLE_model_str.txt', models, query_inds)

    # Plot model results
    plt.rcParams['figure.figsize'] = [8, 5]
    plt.figure()
    k=0
    for i in sorted_inds_[1:]:
        plt.plot(z, eval(models['str_'][i]), color='grey', label=k and '_nolegend_' or r'$M_i:$ Ensemble of all models')
        k=1
    plt.plot(z, f_target, 'o-', color='k', markevery=7, label='$M_{target} = $' + hidden_model_str)
    plt.plot(z, eval(models['str_'][best_ind_]), label='$M_{best}   = $' + models['str_'][best_ind_])
    plt.xlabel('$z$')
    plt.legend()
    plt.savefig('data/fig/SIMPLE_model_result.png', bbox_inches='tight')

    print('Hidden model :', hidden_model_str)
    print('Best model   :', models['str_'][best_ind_])


