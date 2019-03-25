"""
Collection of utility functions for plotting and output of model strings.

2019, M.Schmelzer
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sympy as sy


def init_plotting():
    print('Adjusting style!')
    plt.style.use('ggplot')
    # no background grid
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.facecolor'] = 'w'
    # change greyish to black
    plt.rcParams['axes.edgecolor'] = 'k'  # u'#555555'
    plt.rcParams['axes.labelcolor'] = 'k'
    plt.rcParams['xtick.color'] = 'k'
    plt.rcParams['ytick.color'] = 'k'
    plt.rcParams['legend.facecolor'] = 'w'
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = 10,7.5


def _init_plotting(case):
    # plt.rcParams['image.cmap'] = 'jet'#'grey'
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Ubuntu'
    # plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 15
    # plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['lines.markersize'] = 14
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams['figure.figsize'] = 5, 5  # 10,7.5
    # plt.rcParams['figure.titlesize'] = 14a


def c():
    plt.close('all')


def get_colors(plot_test=False, style='ggplot', print_all_styles=False):
    if print_all_styles:
        print(plt.style.available)
        plt.style.use(style)

    c = []
    for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
        c.append(color['color'])

    if plot_test:
        a = range(len(c))
        plt.figure()
        for i in range(len(c)):
            plt.plot(a[i], 1, 'o', color=c[i])
            plt.text(a[i], 1 * 1.01, 'i=' + str(i))

    print("c = (0:red, 1:blue, 2:violet, 3:grey, 4:yellow, 5:green, 6:light red)")
    return c


def get_colormap(which_colormap='c', n_bin=21):
    colors = {'a': [np.array([0.20651241, 0.37553937, 0.58569048, 1.]),
                    np.array([0.18728497, 0.31190092, 0.46683585, 1.]),
                    np.array([0.16805752, 0.24826247, 0.34798123, 1.]),
                    np.array([0.14883008, 0.18462402, 0.22912661, 1.]),
                    np.array([0.133, 0.133, 0.133, .6]),
                    np.array([0.24823135, 0.15207457, 0.14236809, 1.]),
                    np.array([0.38569861, 0.17449712, 0.15317753, 1.]),
                    np.array([0.52316588, 0.19691968, 0.16398697, 1.]),
                    np.array([0.65652964, 0.2186729, 0.17447374, 1.])],
              'b': [np.array([0.24715576, 0.49918708, 0.57655991, 1.]),
                    np.array([0.54439601, 0.70003822, 0.74781989, 1.]),
                    np.array([0.133, 0.133, 0.133, .5]),
                    np.array([0.85649414, 0.5955863, 0.52926605, 1.]),
                    np.array([0.7634747, 0.33484566, 0.2258923, 1.])],
              'c': [(0.32860444731764704, 0.43971182997647057, 0.8695872625411765),
                    (0.4358148063058824, 0.5707073031529412, 0.951717381282353),
                    (0.5543118699137254, 0.6900970112156862, 0.9955155482352941),
                    # (0.6672529243333334, 0.7791764569999999, 0.992959213),
                    # (0.7727059486039215, 0.8389782172392156, 0.9493187599137255),
                    (0.6, 0.6, 0.6),
                    # (0.9383263563333333, 0.8089165520313726, 0.741161515027451),
                    # (0.968203399, 0.7208441, 0.6122929913333334),
                    (0.9566532109764706, 0.598033822717647, 0.4773022923529412),
                    (0.9057834780117647, 0.4551856921647059, 0.35533588384705883),
                    (0.8204010983882353, 0.2867649126352941, 0.2451595198)],
              'd': [(0.32860444731764704, 0.43971182997647057, 0.8695872625411765),
                    (0.4358148063058824, 0.5707073031529412, 0.951717381282353),
                    (0.5543118699137254, 0.6900970112156862, 0.9955155482352941),
                    (0.6672529243333334, 0.7791764569999999, 0.992959213),
                    (0.7727059486039215, 0.8389782172392156, 0.9493187599137255),
                    (0.6, 0.6, 0.6),
                    (0.9383263563333333, 0.8089165520313726, 0.741161515027451),
                    (0.968203399, 0.7208441, 0.6122929913333334),
                    (0.9566532109764706, 0.598033822717647, 0.4773022923529412),
                    (0.9057834780117647, 0.4551856921647059, 0.35533588384705883),
                    (0.8204010983882353, 0.2867649126352941, 0.2451595198)]}
    # sns.diverging_palette(255, 133, l=60, n=7, center="dark", as_cmap=True)
    return matplotlib.colors.LinearSegmentedColormap.from_list(which_colormap, colors[which_colormap], N=n_bin)


def write_model_to_latex(term):
    if term=='const':
        tmp = sy.sympify(term.replace('**1.0', '').replace('np.cos', 'cos').replace('np.sin', 'sin'))
    else:
        tmp = sy.sympify(term.replace('**1.0', '').replace('np.cos', 'cos').replace('np.sin', 'sin')).replace('const',
                                                                                                              '1')
    model_str = '$' + sy.latex(tmp) + '$'
    return model_str


def write_models_to_txt(file_name, models, query_inds=None):
    if query_inds is None:
        query_inds = models.index.tolist()
    with open(file_name, 'w') as f:
        for i in query_inds:
            f.write('###\n')
            f.write('index: ' + str(i) + '\n')
            f.write('MSE  = ' + str(models['mse_'][i]) + '\n')
            f.write('model: ' + str(models['str_'][i]) + '\n')
            f.write('latex: ' + write_model_to_latex(models['str_'][i]) + '\n\n')


def get_labels(lib):
    return np.array([write_model_to_latex(np.array(lib)[i]) for i in range(len(lib))])


def plot_matrix(model_mat, library, gs, ax_ind=2):
    labels = get_labels(library)

    cmap = get_colormap(which_colormap='c')
    cmap.set_bad(color='white')

    ax = plt.subplot(gs[ax_ind])
    model_mat = np.ma.masked_where(model_mat == 0.0, model_mat)
    #    ax2.set_xticks([])#ax2.set_xticks(np.arange(0,len(MSE_vec),1))#
    #    ax0.set_xticklabels(all_labels, rotation=90)
    ax.tick_params(which='minor', length=0)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True);

    ax.set_yticklabels(np.arange(1, model_mat.shape[0] + 1))
    ax.set_yticks(np.arange(0, model_mat.shape[0], 1))
    ax.set_yticks(np.arange(-.5, model_mat.shape[0], 1), minor=True)

    if model_mat.shape[0] > 15:
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)

    ax.set_ylabel('model index $\; i$')
    r = np.max(abs(model_mat))
    # tmp = model_mat[ind_sort_y, :]
    im = ax.imshow(model_mat[:, :], cmap=cmap, aspect='auto', origin='lower', vmin=-r, vmax=r)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    return im


def plot_model_and_mse(models, lib, inds):
    model_mat = np.stack(models['coef_'][inds].values)
    num_models = model_mat.shape[0]
    num_active_cand = model_mat.shape[1]

    # Sort according to complexity
    ind_sort = np.array([np.count_nonzero(model_mat[::-1, :][:, i]) for i in range(num_active_cand)]).argsort()[::-1]

    plt.figure(figsize=(8, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[10, 1.5], height_ratios=[1, 15])
    gs.update(wspace=0.05, hspace=0.05)

    # Plot model matrix
    im = plot_matrix(model_mat[:, ind_sort], lib[ind_sort], gs)

    # Plot colorbar
    ax0 = plt.subplot(gs[0])
    cb = plt.colorbar(im, cax=ax0, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    # Plot MSE values
    ax3 = plt.subplot(gs[3])
    ax3.plot(models['mse_'][inds].values, np.arange(0, num_models), 'o', label='')
    xticks = ax3.get_xticks()
    ax3.set_xticks([xticks[0], xticks[-1]], minor=True)
    if num_models > 15:
        ax3.set_yticks(np.arange(0, num_models, 1))
    ax3.set_ylim([-0.5, num_models-0.5])
    ax3.set_yticklabels([])
    ax3.set_xlabel('MSE')





