import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.io as sio
from numpy.linalg import lstsq
import sympy as sym
import time as t

sys.path.append(os.path.abspath(
   "/home/administrateur/Documents/lmfl_data/Programmes/python_codes/langevin-regression"))

import utils_reg as utils
import fpsolve_reg as fpsolve
import functions_regression as fr

import netCDF4 as nc
from kramersmoyal import km, kernels

### Parameters ###
type_method = 'tip'
type_file = "2.4"

N_start = 1
N_end = 10
N_run = N_end - N_start + 1

dimt = 2000

path = "/home/administrateur/Documents/lmfl_data/Time_distributions_" + \
   type_method + "_" + type_file + "/"
path_save = "/home/administrateur/Documents/lmfl_data/Figures/Essais_modeles/"
file_save = "Cost_evol_spars"

N_bins = 32  # KM bins
stride = 1  # How many snapshots to skip

f_sample = 4
delta_t = 0.25
bw = 1e-6  # Band width for kernels

powers = np.array([[1], [2]])
Coeffs_poly = [5, 4]

Xi = [[2.24652236e-02, 1.99615390e-02, 1.95023594e-02, 2.33196858e-02,
       2.32541207e-02, 2.08580716e-02, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00],
      [2.38793129e+00, 2.06264850e+00, 2.10111157e+00, 2.41446820e+00,
       2.38853559e+00, 2.23371153e+00, 2.07114610e+00, 1.87716848e+00,
       1.26927093e+00, -4.41466544e-01],
      [-7.17401722e+01, -6.63199091e+01, -6.71445137e+01, -6.95747116e+01,
       -6.98968641e+01, -6.33957901e+01, -6.71794843e+01, -7.00631286e+01,
       0.00000000e+00, 0.00000000e+00],
      [-5.26631019e+03, -4.91413805e+03, -5.24427683e+03, -5.15544380e+03,
       -5.15287443e+03, -4.76303335e+03, -4.88062629e+03, -5.12309472e+03,
       -3.79160427e+03, 0.00000000e+00],
      [4.70012107e+04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00],
      [2.18201637e+06, 1.71064623e+06, 1.71924862e+06, 1.32143357e+06,
       1.32741558e+06, 1.18990399e+06, 1.09003957e+06, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00],
      [5.10958320e-04, 5.53863938e-04, 5.73688381e-04, 3.90546591e-04,
       1.25746662e-03, 1.64356759e-03, 1.45471464e-03, 1.43313648e-03,
       -4.91112633e-07, -5.91353188e-07],
      [-3.72884222e-03, -3.47352593e-03, 1.69963547e-03, 3.36829240e-03,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00],
      [-2.03524948e+00, -1.82109661e+00, -4.26808915e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00],
      [1.07489423e+00, 3.67995731e+00, 6.24495324e+00, 4.94493468e+00,
       -2.42896035e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00],
      [9.86202941e+02, 1.16733478e+03, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00]]

Xi = np.array(list(Xi), dtype=float)

V = np.array([3.03596336e+01, 4.70017060e-02, 4.37445148e-02, 4.23958639e-02, 4.23854683e-02,
             4.20954367e-02, 5.67211587e-02, 5.13871057e-02, 2.99010005e+01, 5.15204635e+01])

### Build KM coeffs // Build Sympy ###

KMc = fr.KM_list()
fun_expr = []

for i in range(len(powers)):
   fun_expr.append(fr.build_sympy_poly(Coeffs_poly[i]))

labels = [r'${0}$'.format(sym.latex(t)) for t in np.concatenate(fun_expr)]

active = np.abs(Xi) > 1e-8

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

plt.rc('font', **font)

n_terms = len(labels)
fg_ov_fit = plt.figure(figsize=(18, 8))
plt.scatter(np.arange(len(V)), V, c='k', s=200)
plt.gca().set_xticks(np.arange(n_terms-1))
plt.gca().set_xticklabels(np.arange(n_terms, 1, -1))
plt.xlabel('Number of non-zero terms')
plt.ylabel(r'Cost')
#plt.gca().set_yscale('log')
plt.grid()

fg_spars = plt.figure(figsize=(12, 4))
plt.pcolor(active, cmap='bone_r', edgecolors='gray')
plt.gca().set_yticks(0.5+np.arange(n_terms))
plt.gca().set_yticklabels(labels)
plt.gca().set_xticks(0.5+np.arange(n_terms-1))
plt.gca().set_xticklabels(np.arange(n_terms, 1, -1))
plt.xlabel('Sparsity')
plt.ylabel('Active terms')
plt.show()

### Save figures ###
fg_ov_fit.savefig(path_save + file_save + '.eps', format='eps')
fg_ov_fit.savefig(path_save + file_save + '.png', format='png')
