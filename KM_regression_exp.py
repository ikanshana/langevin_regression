import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sys
import os
import scipy.io as sio
from numpy.linalg import lstsq
import time as t

sys.path.append(os.path.abspath(
	"/home/administrateur/Documents/lmfl_data/Programmes/python_codes/langevin-regression"))

import utils
import fpsolve

### Parameters ###
type_method = 'tip'
type_file = '2.4'

N_start = 1
N_end = 10
N_run = N_end - N_start + 1

dimt = 2000
delta_t = 0.25  # Exp sampling

N_bins = 32  # KM bins
stride = 1  # How many snapshots to skip

path = "/home/administrateur/Documents/lmfl_data/Time_distributions_" + \
	type_method + "_" + type_file + "/"

path_save = "/home/administrateur/Documents/lmfl_data/Figures/Essais_modele/"
file_save = "autocorr_Markov_" + type_file + "_" + type_method

### LOAD DATA ###
Ym_tot = np.zeros([N_run*dimt, 1])

for run in range(N_start, N_end + 1):
	file_data = "time_distribution_run" + f'{run:02d}' + ".mat"

	data = sio.loadmat(path + file_data)

	Ym_tot[(run - N_start)*dimt:(run - N_start + 1)*dimt] = data['Ym']

### COMPUTE KM ###
edges = np.squeeze(np.linspace(min(Ym_tot), max(Ym_tot), N_bins + 1))
dx = edges[1] - edges[0]
centers = (edges[:-1]+edges[1:])/2

f_KM, a_KM, f_err, a_err = utils.KM_avg(
	Ym_tot, edges, stride=stride, dt=delta_t)

### BUILD SYMPY ###
x = sym.symbols('x')

f_expr = np.array([x**i for i in np.arange(6)])  # Polynomial library for drift
# Polynomial library for diffusion
s_expr = np.array([x**i for i in np.arange(5)])

print(f_expr)

lib_f = np.zeros([len(f_expr), N_bins])

for k in range(len(f_expr)):
	lamb_expr = sym.lambdify(x, f_expr[k])
	lib_f[k] = np.squeeze(lamb_expr(centers))

lib_s = np.zeros([len(s_expr), N_bins])
for k in range(len(s_expr)):
	lamb_expr = sym.lambdify(x, s_expr[k])
	lib_s[k] = np.squeeze(lamb_expr(centers))

### INITIALIZE Xi ###
Xi0 = np.zeros((len(f_expr) + len(s_expr)))
mask = np.nonzero(np.isfinite(f_KM))[0]

Xi0[:len(f_expr)] = lstsq(lib_f[:, mask].T, f_KM[mask],
                          rcond=None)[0]   # Regression against drift
Xi0[len(f_expr):] = lstsq(lib_s[:, mask].T, np.sqrt(
	2*a_KM[mask]), rcond=None)[0]  # Regression against diffusion

print('Value of Xi0 :')
print(Xi0)

### COMPUTE WEIGHTS ###
# Not that necessary
f_err = np.ones(N_bins)
a_err = np.ones(N_bins)

#f_err[:12] = 1
#f_err[12:-13] = 1./(f_KM[12:-13]/np.mean(f_KM[12:-13]))**6
#f_err[-13:] = 1

W = np.array((f_err.flatten(), a_err.flatten()))
# Set zero entries to large weights
W[np.less(abs(W), 1e-12, where=np.isfinite(W))] = 1e6
# Set NaN entries to large numbers (small weights)
W[np.logical_not(np.isfinite(W))] = 1e6
W = 1/W  # Invert error for weights
W = W/np.nansum(W.flatten())

W = np.ones(W.shape)

### OPTIMISATION ###
# Compute empirical PDF
p_hist = np.histogram(Ym_tot, edges, density=True)[0]

# Initialize adjoint solver
afp = fpsolve.AdjFP(centers)

# Initialize forward steady-state solver
fp = fpsolve.SteadyFP(N_bins, dx)

# Optimization parameters
params = {"W": W, "f_KM": f_KM, "a_KM": a_KM, "Xi0": Xi0,
          "f_expr": f_expr, "s_expr": s_expr,
          "lib_f": lib_f.T, "lib_s": lib_s.T, "N": N_bins,
          "kl_reg": 10,
          "fp": fp, "afp": afp, "p_hist": p_hist, "tau": stride*delta_t,
          "radial": False}

# Optimisation


def opt_fun_local(params): return utils.AFP_opt(utils.cost, params)


Xi, V = utils.SSR_loop(opt_fun_local, params)

####################
# SSR cost function
####################

labels = [r'${0}$'.format(sym.latex(t))
          for t in np.concatenate((f_expr, s_expr))]

active = abs(Xi) > 1e-8

n_terms = len(labels)
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(np.arange(len(V)), V, c='k')
plt.gca().set_xticks(np.arange(n_terms-1))
plt.gca().set_xticklabels(np.arange(n_terms, 1, -1))
plt.xlabel('Sparsity')
plt.ylabel(r'Cost')
#plt.gca().set_yscale('log')
plt.grid()

plt.subplot(122)
plt.pcolor(active, cmap='bone_r', edgecolors='gray')
plt.gca().set_yticks(0.5+np.arange(n_terms))
plt.gca().set_yticklabels(labels)
plt.gca().set_xticks(0.5+np.arange(n_terms-1))
plt.gca().set_xticklabels(np.arange(n_terms, 1, -1))
plt.xlabel('Sparsity')
plt.ylabel('Active terms')
plt.show()
plt.ylabel(r'Cost')
#plt.gca().set_yscale('log')
plt.grid()

plt.subplot(122)
plt.pcolor(active, cmap='bone_r', edgecolors='gray')
plt.gca().set_yticks(0.5+np.arange(n_terms))
plt.gca().set_yticklabels(labels)
plt.gca().set_xticks(0.5+np.arange(n_terms-1))
plt.gca().set_xticklabels(np.arange(n_terms, 1, -1))
plt.xlabel('Sparsity')
plt.ylabel('Active terms')
plt.show()
