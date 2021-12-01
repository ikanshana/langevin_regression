import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.io as sio
from numpy.linalg import lstsq
import sympy as sym
import time as t

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

path = "/workdir/indra.kanshana/2bar_project/data/Time_distributions_" + \
    type_method + "_" + type_file + "/"

N_bins = 32  # KM bins
stride = 1  # How many snapshots to skip

f_sample = 4
delta_t = 0.25
bw = 1e-6  # Band width for kernels

powers = np.array([[1], [2]])
Coeffs_poly = [5, 4]

dt = delta_t/1e4

save_load = "/home/administrateur/Documents/lmfl_data/Regression_results/save_step.nc"
checkpoint_load = False

### Load data ###
Ym_tot = np.zeros([N_run*dimt, 1])

for run in range(N_start, N_end + 1):
    file_data = "time_distribution_run" + f'{run:02d}' + ".mat"
    data = sio.loadmat(path + file_data)

    Ym_tot[(run - N_start)*dimt:(run - N_start + 1)*dimt] = data['Ym']

### Compute K-M coeffs + Edges ###
bins = np.array([N_bins])

KMc_exp, Edges = km(Ym_tot, bw=bw, bins=bins, powers=powers)
filtre = KMc_exp == 0.
KMc_exp[filtre] = np.NaN

Edges_X = np.linspace(min(Ym_tot), max(Ym_tot), N_bins + 1)
X_values = (np.diff(Edges_X, axis=0) + 2*Edges_X[:-1])/2

KMc_exp = KMc_exp*f_sample

### Compute histogram ###

p_hist = np.histogramdd(Ym_tot, bins=bins, density=True)[0]

### Build KM coeffs // Build Sympy ###

KMc = fr.KM_list()

f_KM, a_KM, f_err, a_err = utils.KM_avg(
	Ym_tot, Edges_X, stride=stride, dt=delta_t)

KMc_exp[0, :] = f_KM
KMc_exp[1, :] = a_KM

for i in range(len(powers)):
    fun_expr = fr.build_sympy_poly(Coeffs_poly[i])
    coefficient = fr.KM_coefficient(
        powers[i], fun_expr, KMc_exp[i, :].flatten())
    KMc.add_KM_coeff(powers[i], coefficient)

### Compute values Sympy ###
KMc_reg = fr.KM_list(KM_copy=KMc)
KMc_reg.convert_to_langevin_1D()

KMc.compute_sym_values([X_values])
KMc_reg.compute_sym_values([X_values])

### Initialise Xi ###
N = sum(Coeffs_poly) + 2
Xi0 = np.zeros(N)
size_tot = 0

for k in range(len(powers)):
    fit_KM = KMc_reg.get_fun_value(k)
    Flat_KM = KMc_reg.get_exp_value(k)
    mask = np.nonzero(np.isfinite(Flat_KM))
    Xi0[size_tot:size_tot + len(fit_KM)] = lstsq(
        np.squeeze(fit_KM[:, mask]).T, Flat_KM[mask], rcond=None)[0]

    size_tot += len(fit_KM)

print('Value of Xi0 :')
print(Xi0)

### COMPUTE WEIGHTS ###

### OPTIMISATION ###
#Compute empirical PDF
W = np.ones((len(powers), N_bins))
#Initialise adjoint solver
afp = fpsolve.AdjFP(np.array((X_values)), ndim=1, solve='exp', dt=dt)

#Initialise forward steady-state solver
dx = np.array((Edges_X[1] - Edges_X[0]))
fp = fpsolve.SteadyFP(N_bins, dx)

#Optimisation parameters
params = {"W": W, "KMc": KMc_reg, "Xi0": Xi0, "N": N_bins, "p_hist": p_hist,
          "kl_reg": 10, "Nbr_iter_max": 1e5, "track": 0,
          "fp": fp, "afp": afp, "tau": stride*delta_t, "radial": False,
          "print_cost": True, "checkpoint_file": save_load,
          "checkpoint_load": checkpoint_load}


def opt_fun_local(params): return utils.AFP_opt(utils.cost_reg, params)


Xi, V = utils.SSR_loop(opt_fun_local, params)

np.save("weights/Xi_1D_exp", Xi)
np.save("weights/V_1D_exp", V)

#def opt_fun_local(params): return utils.AFP_opt(utils.cost_reg, params)

#Xi, V = utils.SSR_loop(opt_fun_local, params)
