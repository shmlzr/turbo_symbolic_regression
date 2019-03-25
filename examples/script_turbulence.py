"""
====================================================================================
Script to perform symbolic regression on turbulence data
====================================================================================
"""
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
import turbosparsereg.dataprocessing.dataprocessor as dataproc
from turbosparsereg.util import util
from turbosparsereg.methods.sr_elasticnet import TSRElasticNet


def process_turbulence_data(case, case_params, do_process_turbulence_data=False):
    """ Load and retrieve data of turbulent flow from high-fidelity simulations case.
    """
    if do_process_turbulence_data:
        frozen = pickle.load(open('data/' + case + '_frozen_var' + '.p', 'rb'))
        data_i = frozen['data_i']

        # Calculate strain and rotation rate tensor
        Sij, Wij, timescale_norm = dataproc.calc_sij_wij(data_i['grad_u'], data_i['omega_frozen'])

        # Compute tensor basis and its invariants
        T = dataproc.calc_tensor_basis(Sij, Wij)
        Inv = dataproc.calc_invariants(Sij, Wij)

        # Calc additional flow features
        features = dataproc.calc_flow_features(data_i['d'], data_i['k'], 1. / eval(case_params[case]['Re']))

        # Compute discrepancy stress as target
        aDelta, bDelta = dataproc.compute_target(data_i['aij'], Sij, data_i['k'],
                                                 data_i['nut_frozen'], data_i['omega_frozen'])

        # Compute s=T:grad_u
        s = dataproc.calc_s(T, data_i['grad_u'])

        # Store all variables in dict
        data_i.update({'Sij': Sij, 'Wij': Wij, 'timescale_norm': timescale_norm,
                       'T': T, 'Inv': Inv, 'features': features,
                       'aDelta': aDelta, 'bDelta': bDelta, 's': s})

        # Flattening tensors fields to long vectors
        data_i.update(zip(['T_flat', 'Inv_flat', 'bDelta_flat', 'features_flat'],
                          dataproc.compose_flat_vectors(data_i['T'], data_i['Inv'],
                                                        data_i['bDelta'], data_i['features'])))

        pickle.dump(data_i, open('data/data_i.p', 'wb'))

    else:
        data_i = pickle.load(open('data/data_i.p', 'rb'))

    return data_i


def create_library(case, data_i, active, operations, do_create_library=False):
    """ Helper function to create library of candidate functions
    """
    var_dict = dataproc.declare_symbols(data_i['Inv_flat'],
                                        data_i['T_flat'],
                                        data_i['Wij'],
                                        data_i['features_flat'])

    var_dict_full = dataproc.declare_symbols(data_i['Inv'],
                                             data_i['T'],
                                             data_i['Wij'],
                                             data_i['features'])

    # Instantiate BuildLibrary
    buildlib = dataproc.BuildLibrary(operations)

    if do_create_library:
        # Create library of symbolic candidates B
        B = buildlib.build_B_library(var_dict, active)

        # Create library of symbolic candidates C and evaluated candidates C_data
        C, C_data = buildlib.build_and_eval_C_library(B, data_i['T_flat'], var_dict, var_dict_full,
                                                      active, mode='aDelta')

        # Reduce multicollinearity within the library
        C, C_data = buildlib.reduce_multicollinearity(C, C_data, data_i['bDelta_flat'], mode='vif')

        # Store all symbolic and numeric variables in lib
        lib = {'B': B, 'C': C, 'C_data': C_data, 'var_dict': var_dict, 'var_dict_full': var_dict_full}
        pickle.dump(lib, open('data/candidate_libraries_' + case + '_C_var' + active['var'] +
                              '_exp' + active['exp'] + '_fun' + active['fun'] + '.p', 'wb'))

    else:
        lib = pickle.load(open('data/candidate_libraries_' + case + '_C_var' + active['var'] +
                               '_exp' + active['exp'] + '_fun' + active['fun'] + '.p', 'rb'))

    return lib


def plot_pcolor_data(data_i, model_uu, label):
    plt.rcParams['figure.figsize'] = [6, 5]
    plt.figure()

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 15, 15])
    # gs.update(wspace=0.05, hspace=0.05)
    ax1 = plt.subplot(gs[1])
    im1 = ax1.pcolor(data_i['meshRANS'][0, :].reshape(120, 130, order='F'),
                     data_i['meshRANS'][1, :].reshape(120, 130, order='F'),
                     data_i['uu'].reshape(120, 130, order='F'), vmin=-0.05, vmax=0.05)
    ax1.set_ylabel('Data ' + label)

    ax0 = plt.subplot(gs[0])
    cb = plt.colorbar(im1, cax=ax0, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    ax2 = plt.subplot(gs[2])
    ax2.pcolor(data_i['meshRANS'][0, :].reshape(120, 130, order='F'),
               data_i['meshRANS'][1, :].reshape(120, 130, order='F'),
               model_uu.reshape(120, 130, order='F'), vmin=-0.05, vmax=0.05)
    ax2.set_ylabel('Model ' + label)


def default_parameters():
    # Name of flow case
    case = 'PH10595'  # Flow over Periodic Hills at Re=10595

    # Dict of case dependent parameters
    case_params = {'PH10595': {'flow_case': 'PeriodicHills',
                               'turb_model': 'kOmegaSST',
                               'Re': '10595', 'num_of_cells': 15600,
                               'nx': 120, 'ny': 130,
                               'x_var': 'x', 'y_var': 'y', 'x_ind': 0, 'y_ind': 1,
                               'end_time': 10000}}

    operations = {'base_tensors': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
                  'var': ['const', 'l1', 'l2', 'l3', 'l4', 'l5', 'm1'],
                  'exp': ['**-3.0', '**-2.0', '**-1.0', '**-0.5', '**0.5', '**1.0', '**2.0', '**3.0'],
                  'fun': ['abs', 'np.log10', 'np.exp', 'np.sin', 'np.cos']}

    # Which raw inputs should be used for the library
    active = {'base_tensors': '11110', 'var': '1110000', 'exp': '00000110', 'fun': '00000'}

    # Discrete structure of elastic net (at which value pairs l1/alpha should the optimization problem be solved)
    n_alphas = 100
    l1_ratios = [.01, .1, .2, .5, .7, .9, .95, .99, 1.0]

    # Set of ridge parameters for post-selection model inference
    ridge_alphas = [0.01, 1e-5]

    return case, case_params, operations, active, n_alphas, l1_ratios, ridge_alphas


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Define what needs to be executed or loaded: If False the data is loaded from /examples/data/
    do_process_turbulence_data = True
    do_create_library          = True

    # ------------------------------------------------------------------------------------------------------------------
    # Define parametric setup for data processing and symbolic regression
    # ------------------------------------------------------------------------------------------------------------------
    case, case_params, operations, active, n_alphas, l1_ratios, ridge_alphas = default_parameters()

    # ------------------------------------------------------------------------------------------------------------------
    # load variables from high-fidelity data set
    # ------------------------------------------------------------------------------------------------------------------
    data_i = process_turbulence_data(case, case_params, do_process_turbulence_data=do_process_turbulence_data)

    # ------------------------------------------------------------------------------------------------------------------
    # Create library of candidate functions
    # ------------------------------------------------------------------------------------------------------------------
    lib = create_library(case, data_i, active, operations, do_create_library)

    # ------------------------------------------------------------------------------------------------------------------
    # Perform sparse regression using ElasticNet
    # ------------------------------------------------------------------------------------------------------------------
    # Define generic variable names and subsample
    sub_sampler = 50 #Subsampling data to increase speed of regression
    X = lib['C_data'][::sub_sampler, :]
    y = data_i['bDelta_flat'][::sub_sampler]

    # Train/test split of the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Perform discovery of model form (Which features are active?)
    tsr = TSRElasticNet(n_alphas=n_alphas, l1_ratios=l1_ratios, ridge_alphas=ridge_alphas).fit(x_train, y_train)

    # Perform model inference/calibration (Compute values of active coefficients)
    models = tsr.model_inference(x_train, x_test, y_train, y_test, lib['C'])

    # ------------------------------------------------------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------------------------------------------------------
    ##
    plt.close('all')
    util.init_plotting()

    # Extract models of a specific tuning parameter ridge_alpha
    query_inds = models.query("ridge_alpha==1e-5").index
    best_ind_ = models['mse_'].idxmin()

    # plot model matrix vs mean squared error
    util.plot_model_and_mse(models, lib['C'], query_inds)
    plt.savefig('fig/TURBULENCE_model_vs_mse.png', bbox_inches='tight')

    # Write out results in model_str.txt
    util.write_models_to_txt('models/TURBULENCE_model_str.txt', models, query_inds)

    # Plot uu component
    model_uu = dataproc.eval_model(models['str_'][best_ind_], lib['var_dict_full'])[0, 0, :]
    plot_pcolor_data(data_i, model_uu, '$uu$')
    plt.savefig('fig/TURBULENCE_model_result.png', bbox_inches='tight')

    print('Best model   :', models['str_'][best_ind_])
