"""
Load and retrieve data of turbulent flows.

2019, M.Schmelzer
"""
import numpy as np
import numpy.ma as ma
import sympy as sy
import pandas as pd
import scipy.signal as signal
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.preprocessing as preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor


def filter_data(data, nx, ny, type_var='tensor', kernel_size=3):
    """ Filters data using a median filter with kernel size 3 (default).

    :param data: input field to be filtered
    :param nx: num of mesh points in x direction
    :param ny: num of mesh points in y direction
    :param type_var: type of field (scalar, tensor)
    :param kernel_size: size of the kernel filter
    :return: flatted filtered field
    """
    output = 0
    if type_var == 'tensor':
        output = np.zeros(shape=data.shape)
        tmp = get_plane(data, nx, ny, 'tensor')
        for i in range(3):
            for j in range(3):
                output[i, j, :] = signal.medfilt2d(tmp[i, j, :, :], kernel_size=kernel_size).flatten(order='F')

    elif type_var == 'scalar':
        tmp = data.reshape(nx, ny, order='F')
        output = signal.medfilt2d(tmp[:, :], kernel_size=kernel_size).flatten(order='F')

    return output


def calc_eigen_values(bij):
    """ Calculates the eigenvalues of a tensor field.

    :param bij: tensor field (3. 3, num_of_points)
    :return: eigenvalues (3, num_of_points)
    """
    length = bij.shape[2]
    eigenvalues = np.zeros([3, length])
    for i in range(length):
        try:
            a, b = np.linalg.eig(bij[:, :, i])
            eigenvalues[:, i] = sorted(a, reverse=True)
        except np.linalg.LinAlgError:
            eigenvalues[:, i] = np.ones(3) * np.nan
    return eigenvalues


def mean_absolute_percentage_error(y_true, y_pred):
    """ Compute the mean absolute percentage error.

    :param y_true: target
    :param y_pred: prediction
    :return: mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    abs_error = [np.abs((y_true[i] - y_pred[i]) / y_true[i]) for i in range(len(y_true)) if y_true[i]]
    return np.mean(abs_error) * 100


def dirac(num_points):
    """ Builds a Dirac function field.
    The Dirac function \delta_ij is 1 on each component fo the diagonal of a tensor and zero elsewhere.

    :param num_points: number of cells
    :return: Dirac function field
    """
    return np.asarray([np.eye(3)] * num_points).T


def calc_sij_wij(grad_u, omega, no_dim=True):
    """ Calculates the strain rate and rotation rate tensors.  Normalizes by omega:
    Sij = k/eps * 0.5* (grad_u  + grad_u^T)
    Wij = k/eps * 0.5* (grad_u  - grad_u^T)
    :param grad_u: velocity gradient (3, 3, num_of_points)
    :return: Sij, Wij (3, 3, num_of_points), timescale (num_of_points)
    """
    omega = np.maximum(omega, 1e-8)

    if no_dim:
        timescale = 1 / omega[:]
    else:
        timescale = np.ones(shape=omega.squeeze().shape)

    num_of_cells = grad_u.shape[2]
    Sij = np.zeros((3, 3, num_of_cells))
    Wij = np.zeros((3, 3, num_of_cells))
    timescale_norm = np.zeros(num_of_cells)

    for i in range(num_of_cells):
        tmp = 0.5 * (grad_u[:, :, i] + np.transpose(grad_u[:, :, i]))
        timescale_norm[i] = 1. / max(np.sqrt(2 * np.tensordot(tmp, tmp)) / 0.31, omega[i])
        # timescale_norm[i] = np.sqrt( (k[i]/epsilon[i])**2.0 + 2/Re_T[i] )
        Sij[:, :, i] = timescale_norm[i] * 0.5 * (grad_u[:, :, i] + np.transpose(grad_u[:, :, i]))
        Wij[:, :, i] = timescale_norm[i] * 0.5 * (grad_u[:, :, i] - np.transpose(grad_u[:, :, i]))

    for i in range(num_of_cells):
        Sij[:, :, i] = Sij[:, :, i] - 1. / 3. * np.eye(3) * np.trace(Sij[:, :, i])

    return Sij, Wij, timescale_norm


def compute_target(aij, Sij, k, nut_frozen, omega_frozen):
    """ Compute discrepancy stress aDelta and bDelta.

    :param aij: Anisotropic Reynolds stress (3, 3, num_of_points)
    :param Sij: Mean strain-rate tensor (3, 3, num_of_points)
    :param k: Turbulent kinetic energy (num_of_points)
    :param nut_frozen: Frozen eddy viscosity (num_of_points)
    :param omega_frozen: Frozen specific dissipation rate (num_of_points)
    :return: aDelta, bDelta: Discrepancy stress and normalised discrepancy stress
    """
    # aDelta = tauDNS + 2*nut*omega*Sij - 2./3 * dirac * k
    aDelta = aij + 2 * nut_frozen * omega_frozen * Sij  # -2*nut_frozen*omega_frozen*Sij #
    bDelta = aDelta / (2 * k)
    return aDelta, bDelta


def calc_tensor_basis(Sij, Wij):
    """ Calculates the integrity basis of the base tensor expansion.

    :param Sij: Mean strain-rate tensor (3, 3, num_of_points)
    :param Wij: Mean rotation-rate tensor (3, 3, num_of_points)
    :return: T: Base tensor series (10, 3, 3, num_of_points)
    """
    num_of_cells = Sij.shape[2]
    T = np.ones([10, 3, 3, num_of_cells]) * np.nan
    for i in range(num_of_cells):
        sij = Sij[:, :, i]
        wij = Wij[:, :, i]
        T[0, :, :, i] = sij
        T[1, :, :, i] = np.dot(sij, wij) - np.dot(wij, sij)
        T[2, :, :, i] = np.dot(sij, sij) - 1. / 3. * np.eye(3) * np.trace(np.dot(sij, sij))
        T[3, :, :, i] = np.dot(wij, wij) - 1. / 3. * np.eye(3) * np.trace(np.dot(wij, wij))
        T[4, :, :, i] = np.dot(wij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), wij)
        T[5, :, :, i] = np.dot(wij, np.dot(wij, sij)) \
                        + np.dot(sij, np.dot(wij, wij)) \
                        - 2. / 3. * np.eye(3) * np.trace(np.dot(sij, np.dot(wij, wij)))
        T[6, :, :, i] = np.dot(np.dot(wij, sij), np.dot(wij, wij)) - np.dot(np.dot(wij, wij), np.dot(sij, wij))
        T[7, :, :, i] = np.dot(np.dot(sij, wij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(wij, sij))
        T[8, :, :, i] = np.dot(np.dot(wij, wij), np.dot(sij, sij)) + np.dot(np.dot(sij, sij), np.dot(wij, wij)) \
                        - 2. / 3. * np.eye(3) * np.trace(np.dot(np.dot(sij, sij), np.dot(wij, wij)))
        T[9, :, :, i] = np.dot(np.dot(wij, np.dot(sij, sij)), np.dot(wij, wij)) \
                        - np.dot(np.dot(wij, np.dot(wij, sij)), np.dot(sij, wij))

        # Enforce zero trace for anisotropy
        for j in range(10):
            T[j, :, :, i] = T[j, :, :, i] - 1. / 3. * np.eye(3) * np.trace(T[j, :, :, i])

    return T


def calc_invariants(Sij, Wij):
    """ Given the non-dimensionalized mean strain rate and mean rotation rate tensors Sij and Rij,
    this returns a set of normalized scalar invariants.

    :param Sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
    :param Wij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
    :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
    """
    num_of_invariants = 5
    num_of_cells = Sij.shape[2]
    invariants = np.zeros((num_of_invariants, 3, 3, num_of_cells))
    # k_tensor = np.zeros([3,3,self.num_of_cells])
    prim_tensor = np.ones([3, 3])
    for i in range(num_of_cells):
        invariants[0, :, :, i] = np.trace(np.dot(Sij[:, :, i], Sij[:, :, i])) * prim_tensor
        invariants[1, :, :, i] = np.trace(np.dot(Wij[:, :, i], Wij[:, :, i])) * prim_tensor
        invariants[2, :, :, i] = np.trace(np.dot(Sij[:, :, i], np.dot(Sij[:, :, i], Sij[:, :, i]))) * prim_tensor
        invariants[3, :, :, i] = np.trace(np.dot(Wij[:, :, i], np.dot(Wij[:, :, i], Sij[:, :, i]))) * prim_tensor
        invariants[4, :, :, i] = np.trace(
            np.dot(np.dot(Wij[:, :, i], Wij[:, :, i]), np.dot(Sij[:, :, i], Sij[:, :, i]))) * prim_tensor

    return invariants


def calc_flow_features(d, k, nu):
    """ Compute additional flow features.

    :param d: Distance from next wall within the domain
    :param k: Turbulent kinetic energy
    :param nu: Physical viscosity
    :return: flow_features: Array of flow features (#features, 3, 3, num_of_points)
    """
    num_of_cells = k.shape[0]
    num_flow_features = 1
    flow_features = np.zeros((num_flow_features, 3, 3, num_of_cells))
    for i in range(num_of_cells):
        tmp = np.sqrt(k[i]) * d[i] / (50 * nu) * np.ones([3, 3])
        tmp[tmp > 2] = 2
        flow_features[0, :, :, i] = tmp
    return flow_features


def calc_Pk(aij, grad_u):
    """ Compute production term from anisotropic stress tensor and velocity gradient tensor.

    :param aij: anisotropic stress tensor (3,3,num_of_points)
    :param grad_u: velocity gradient tensor (3,3,num_of_points)
    :return: Pk: scalar production (num_of_points)
    """
    num_of_cells = grad_u.shape[-1]
    Pk = np.zeros([num_of_cells])
    for i in range(num_of_cells):
        Pk[i] = np.tensordot(aij[:, :, i], grad_u[:, :, i])
    return Pk


def calc_s(T, tensor):
    """Compute double inner dot product between matrix of base tensors and given tensor.

    :param T: Matrix of base tensors (N, 3, 3, num_of_points)
    :param tensor: Given tensor (3, 3, num_of_points)
    :return: s: Double inner dot product
    """
    num_of_cells = tensor.shape[2]
    s = np.zeros([T.shape[0], num_of_cells])
    for n in range(T.shape[0]):
        for i in range(num_of_cells):
            s[n, i] = np.tensordot(T[n, :, :, i], tensor[:, :, i])
    return s


def tensor_flattening(tensor):
    """ Flattens symmetric tensor.

    :param tensor: Given tensor (3,3,num_of_points)
    :return: tensor_flatten: Flatted tensor (6*num_of_points)
    """
    num_of_cells =  tensor.shape[2]
    tensor_flatten = np.zeros([6, num_of_cells])
    tensor_flatten[0, :] = tensor[0, 0, :]
    tensor_flatten[1, :] = tensor[0, 1, :]
    tensor_flatten[2, :] = tensor[0, 2, :]
    tensor_flatten[3, :] = tensor[1, 1, :]
    tensor_flatten[4, :] = tensor[1, 2, :]
    tensor_flatten[5, :] = tensor[2, 2, :]
    return tensor_flatten.flatten()


def tensor_reshaping(tensor_list):
    """ Reshaping flatted tensor.

    :param tensor_list: Flatted tensor array
    :return: tensor (3,3,num_of_points)
    """
    num_of_cells = int(tensor_list.shape[0])
    tensor = np.zeros([3, 3, num_of_cells])
    tensor[0, 0, :] = tensor_list[0, :]
    tensor[0, 1, :] = tensor_list[1, :]
    tensor[0, 2, :] = tensor_list[2, :]
    tensor[1, 0, :] = tensor_list[1, :]
    tensor[1, 1, :] = tensor_list[3, :]
    tensor[1, 2, :] = tensor_list[4, :]
    tensor[2, 0, :] = tensor_list[2, :]
    tensor[2, 1, :] = tensor_list[4, :]
    tensor[2, 2, :] = tensor_list[5, :]
    return tensor


def extract_samples(mesh, var, x_loc, nx, ny, y_ind, var_type='tensor'):
    """ Extract sample profiles from subspace of domain.

    :param mesh: Mesh points
    :param var: Variable to be spatially extracted
    :param x_loc: Location along x axis
    :param nx: Number of mesh points in x direction
    :param ny: Number of mesh points in y direction
    :param y_ind: Indices along y axis to be extracted
    :param var_type: Type of input var
    :return: var_plane: Extracted profile of var
    """
    x_ind = abs(mesh[0, :, 0] - x_loc).argmin()

    if var_type == 'tensor':

        if len(var.shape) == 4:
            num_of_var = var.shape[0]
            var_plane = np.zeros([num_of_var, 3, 3, ny])

            for i in range(num_of_var):
                var_plane[i, :] = get_plane(var[i, :], nx, ny, t='tensor')[:, :, x_ind, y_ind[0]:y_ind[1]]

        elif len(var.shape) == 3:
            var_plane = get_plane(var, nx, ny, t='tensor')[:, :, x_ind, y_ind[0]:y_ind[1]]

    elif var_type == 'scalar':
        var_plane = get_plane(var, nx, ny, t='scalar').squeeze()[x_ind, y_ind[0]:y_ind[1]]

    return var_plane


def trimdata(mesh, var, minmax, var_type='tensor'):
    """ Trims data within spatial domain.

    :param mesh: Mesh points
    :param var: Variable to be spatially extracted
    :param minmax: shape of subdomain
    :param var_type: Type of input var
    :return: trimed data as masked array
    """

    x_min, x_max, y_min, y_max = minmax
    x_logic = np.logical_and(mesh[0, :, :] <= x_max, mesh[0, :, :] >= x_min)
    y_logic = np.logical_and(mesh[1, :, :] <= y_max, mesh[1, :, :] >= y_min)
    nx, ny = mesh.shape[1], mesh.shape[2]

    if var_type == 'tensor':
        var_plane = get_plane(var, nx, ny, t='tensor')

        for i in range(3):

            for j in range(3):
                var_plane[i, j, :] = np.where(x_logic, var_plane[i, j, :], np.nan)
                var_plane[i, j, :] = np.where(y_logic, var_plane[i, j, :], np.nan)

    return ma.masked_where(np.isnan(var_plane), var_plane)


def compose_flat_vectors(T, Inv, bDelta, features):
    """ Flat the tensor input quantities.

    :param T: Base tensors (N,3,3,num_of_points)
    :param Inv: Invariants (M,num_of_points)
    :param bDelta: Target anisotropic Reynolds-stress (3,3,num_of_points)
    :param features: Additional features besides invariants (#features,3,3,num_of_points)
    :return:
    """
    T_flat = np.array([tensor_flattening(T[i, :]) for i in range(10)])
    Inv_flat = np.array([tensor_flattening(Inv[i, :]) for i in range(5)])
    bDelta_flat = tensor_flattening(bDelta).flatten()
    features_flat = np.array([tensor_flattening(features[i, :]) for i in range(features.shape[0])])
    return T_flat, Inv_flat, bDelta_flat, features_flat


def declare_symbols(Inv_flat, T_flat, Wij, features):
    """ Declare a dict with variables {name:data}.

    :param Inv_flat: Flatted invariants
    :param T_flat: Flatted base tensors
    :param Wij: Vorticity rate tensor
    :param features: Additional features
    :return: var_dict
    """
    const = np.ones(Inv_flat[0, :].shape)
    var_dict = {'const': const, 'l1': Inv_flat[0, :],
                'l2': Inv_flat[1, :], 'l3': Inv_flat[2, :], 'l4': Inv_flat[3, :],
                'l5': Inv_flat[4, :], 'T1': T_flat[0, :], 'T2': T_flat[1, :],
                'T3': T_flat[2, :], 'T4': T_flat[3, :], 'T5': T_flat[4, :],
                'Wij': Wij, 'm1': features[0, :]}
    return var_dict


def get_plane(mesh, nx, ny, t='vector'):
    """ Reformates a list of values to a plane.

    :param mesh: Input var
    :param nx: Number of points in x direction
    :param ny: Number of points in y direction
    :param t: Type of input var
    :return: Plane of var
    """
    x = np.zeros([nx, ny])
    y = np.zeros([nx, ny])
    z = np.zeros([nx, ny])

    if t == 'vector':
        for i in range(ny):
            x[:, i] = mesh[0, i*nx:nx+i*nx]
            y[:, i] = mesh[1, i*nx:nx+i*nx]
            z[:, i] = mesh[2, i*nx:nx+i*nx]
        return np.array([x, y, z])

    elif t == 'scalar':
        x = np.squeeze(x)
        for i in range(ny):
            x[:, i] = mesh[i*nx:nx+i*nx]
        return np.array([x])

    elif t == 'tensor':
        out = np.zeros([3, 3, nx, ny])
        for i in range(ny):
            out[:, :, :, i] = mesh[:, :, i*nx:nx+i*nx]
        return out


def eval_model(model_string, var_dict):
    """ Evaluates a model string.
    :param model_string: String of mathematical expression
    :param var_dict: library of features (symbolic and numeric)
    :return: evaluated model string
    """
    locals().update(var_dict)
    model_eval = eval(model_string)
    return model_eval


# -----------------------------------------------------------------------------
class ReduceVIF(BaseEstimator, TransformerMixin):
    """ Transformer to reduce multicollinearity within a feature matrix using
    variance inflation factor.

    Taken from https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class
    """
    def __init__(self, thresh=5, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = preprocessing.Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
            # X = self.imputer.transform(X)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped_inds = []
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            if X.shape[1] > 2:
                vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

                max_vif = max(vif)
                if max_vif > thresh:
                    maxloc = vif.index(max_vif)
                    dropped_inds.append(X.columns[maxloc])
                    print('Dropping ' + str(X.columns[maxloc]) + ' with vif = ' + str(max_vif))
                    X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                    dropped = True
        return X, dropped_inds


# -----------------------------------------------------------------------------
class BuildLibrary(ReduceVIF):
    """ Build a library of candidate functions.

    Parameters
    ----------
    operations : dict
        Variables as raw features and mathematical operations.
    """

    def __init__(self, operations):
        ReduceVIF.__init__(self)
        self.operations = operations

    @staticmethod
    def is_ok(var):
        """ Check if the input contains nan or inf.

        :param var: array
        :return: boolean (True/False)
        """
        if np.isnan(var).any() or np.isinf(var).any():
            return False
        else:
            return True

    @staticmethod
    def define_operation(which_op, expl_op):
        """ Determine operations to be used.

        :param which_op: Active operation encoding
        :param expl_op: All possible operations
        :return: op: Selected operations
        """
        # print('operations', which_op)
        op = []
        for i in range(len(which_op)):
            opi = eval(which_op[i])
            if opi:
                op.append(expl_op[i])
        return op

    def build_B_library(self, var_dict, active):
        """ Creates a library of candidate functions.

        The main concept comes from the FFX algorithm by Trent McConaghy
        (see: FFX: Fast, Scalable, Deterministic Symbolic Regression Technology. In: 2 GENETIC PROGRAMMING THEORY AND
        PRACTICE VI. 2011)

        :param var_dict: Dict of raw input features
        :param active: Active operation encoding
        :return: B_sym: Symbolic library of candidate functions
        """
        print("\nBuild library of candidate functions")

        # specify all variables as single variables
        locals().update(var_dict)

        # Specify variables and mathematical operations to build the library
        var = self.define_operation(active['var'], self.operations['var'])
        exp = self.define_operation(active['exp'], self.operations['exp'])
        fun = self.define_operation(active['fun'], self.operations['fun'])

        # Exclude const from operations
        var_red = [vari for vari in var if not vari == 'const']

        # generate univariate basis
        B1 = []
        for vari in var_red:
            for expi in exp:
                bexp = vari + expi
                if self.is_ok(eval(bexp)):
                    B1.append(bexp)
        B1 = B1

        # apply canonical functions
        B11 = []
        if len(fun) != 0:
            for i in range(len(B1)):
                bi = B1[i]
                for funi in fun:
                    bfun = funi + '(' + bi + ')'
                    print(bfun)
                    if self.is_ok(eval(bfun)):
                        B11.append(bfun)
        B1 = B1 + B11

        # generate interacting-variable basis
        B2 = []
        for i in range(len(B1)):
            bi = B1[i]
            for j in range(i - 1):
                bj = B1[j]
                binter = '(' + bi + ')*(' + bj + ')'
                if self.is_ok(eval(binter)):
                    B2.append(binter)

        B3 = []
        for i in range(len(B2)):
            bi = B2[i]
            for j in range(len(B1)):
                bj = B1[j]
                binter = '(' + bi + ')*(' + bj + ')'
                if self.is_ok(eval(binter)):
                    B3.append(binter)

        Btmp = ['const'] + B1 + B2 + B3

        # symplify bases
        Bsym = []
        for i in range(len(Btmp)):
            tmp = Btmp[i].replace('np.exp', 'exp')
            tmp = tmp.replace('np.log10', 'log10')
            tmp = tmp.replace('np.sin', 'sin')
            tmp = tmp.replace('np.cos', 'cos')
            tmp = str(sy.simplify(sy.sympify(tmp)))
            tmp = tmp.replace('exp', 'np.exp')
            tmp = tmp.replace('log10', 'np.log10')
            tmp = tmp.replace('sin', 'np.sin')
            tmp = tmp.replace('cos', 'np.cos')
            Bsym.append(tmp)

        # check for dublicates
        Bsym = list(set(Bsym))
        loopcount = len(Bsym)
        i = 0
        while i < loopcount:
            if isinstance(eval(Bsym[i]), int):
                del (Bsym[i])  # deletes all constant fields
                loopcount -= 1
            i += 1

        return np.array(Bsym)

    def build_and_eval_C_library(self, B, T_flat, var_dict, var_dict_full, active, grad_u=None, k=None, mode='aDelta',
                                 check_vif=False):
        """ Build tensor-valued library.

        :param B: Symbolic library of candidate functions
        :param T_flat: Base tensor series
        :param var_dict: Dict of raw input features (subsampled)
        :param var_dict_full: Dict of raw input features
        :param active: Active operation encoding
        :param grad_u: Velocity gradient tensor
        :param k: Turbulent kinetic energy
        :param mode: Mode of regression (aDelta or PkDeficit)
        :param check_vif: Reduce multicollinearity using variance inflation factor
        :return: C, C_Data: Symbolic and numeric library
        """
        base_tensors = self.define_operation(active['base_tensors'], self.operations['base_tensors'])

        locals().update(var_dict)

        if mode == 'aDelta':
            num_of_points = T_flat.shape[1]
        elif mode == 'PkDeficit':
            num_of_points = int(T_flat.shape[1] / 6)
            # base_tensors.remove('T1')

        C = []
        for ci in range(len(B)):
            for bti in base_tensors:
                C.append(B[ci] + '*' + bti)

        num_of_cand = len(C)
        C_data = np.zeros([num_of_points, num_of_cand])
        # C_Pk_data = np.zeros([T_flat.shape[1], num_of_cand])
        for i in range(num_of_cand):
            if mode == 'aDelta':
                C_data[:, i] = eval(C[i])
            elif mode == 'PkDeficit':
                C_data[:, i] = calc_Pk(eval_model(C[i], var_dict_full), grad_u) * 2 * k

        # Sanity check of the library
        inds = np.argwhere(np.linalg.norm(C_data, axis=0) < 1e-3).squeeze().tolist()
        inds = sorted(inds, reverse=True)

        print('Number of deleted indices =', len(inds))
        for ind in inds:
            del (C[ind])
            C_data = np.delete(C_data, ind, axis=1)

        print('Number of unique candidates =', len(C))

        return np.array(C), C_data

    def reduce_multicollinearity(self, C, C_data, target, mode='use_vif'):
        """ Reduce the multicollinearity within a library of candidate functions.

        Specify mode
                    vif      - variance inflation factor
                    corrcoef - correlation coefficient

        :param C: Symbolic library of candidate functions
        :param C_data: Numeric library of candidate functions
        :param target: Target field
        :param mode: mode for reduction of multicol
        :return: C, C_Data: Symbolic and numeric library with retained items
        """
        library = C_data.copy()
        scaler = preprocessing.StandardScaler().fit(library)
        library = scaler.transform(library)

        if mode == 'vif':
            print('\nReducing multicollinearity within lib using variance inflation factor')
            library = pd.DataFrame(library)
            X, dropped_inds = self.fit_transform(library, target)
            retained_inds = [fruit for fruit in range(len(C)) if fruit not in dropped_inds]
            print(retained_inds)
        elif mode == 'corrcoef':
            num_cand = len(C)
            corr = np.zeros([num_cand, num_cand])
            for i in range(15):
                for j in range(i):
                    corr[i, j] = np.corrcoef(library[:, i], library[:, j])[0, 1]

            inds = np.argwhere(np.logical_and(abs(corr) > 0.9, abs(corr) < 0.9999))
            retained_inds = np.arange(0, num_cand)
            retained_inds[inds[:, 1]] = inds[:, 0]
            retained_inds = np.unique(retained_inds)

        return np.array(C)[retained_inds], C_data[:, retained_inds]
