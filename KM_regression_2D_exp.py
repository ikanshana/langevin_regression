import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.io as sio
from numpy.linalg import lstsq
import sympy as sym
import time as t

sys.path.append(os.path.abspath("/home/administrateur/Documents/lmfl_data/Programmes/python_codes/langevin-regression"))

import utils_reg as utils
import fpsolve_reg as fpsolve
import functions_regression as fr

import netCDF4 as nc
from kramersmoyal import km,kernels

### Parameters ###
type_file = "1.25"

path_file = "/home/administrateur/Documents/lmfl_data/G_H_" + type_file + "_POD_all/"
name_file = "G_H_" + type_file + "_large_POD_all.nc"

N_bins = 25 #KM bins
stride = 1 #How many snapshots to skip

f_sample = 4
bw = 1e-6 #Band width for kernels

powers = np.array([[1,0], [0,1], [1,1], [2,0], [0,2]])
cross_terms = [True,True,False,False,False]
Coeff_x = [3,3,2,2,2] #Degrees of polynomes
Coeff_y = [3,3,2,2,2]

f_x_threshold = 0.03
f_y_threshold = 0.03

delta_t = 0.25

dt = delta_t/1e4

### Read Exp Data ###
file_nc = nc.Dataset(path_file + name_file,'r')

U_eig = file_nc['eig_vec'][:]

### Compute K-M coeffs ###
bins = np.array([N_bins, N_bins])

KMc_exp, Edges = km(U_eig[:,:2],bw = bw, bins = bins, powers = powers)
filtre = KMc_exp == 0.
KMc_exp[filtre] = np.NaN

### Filter signal to have smoother drift data ###
#f_x,f_y = fr.filter_2D_signal(KMc_exp[0,:,:], KMc_exp[1,:,:],
#                                           f_x_threshold, f_y_threshold,
#                                           kernel_size = 3,
#                                           method = 'localmean',
#                                           max_iter = 3)

#KMc_exp[0] = f_x
#KMc_exp[1] = f_y

### Compute Edges ###

#Edges = np.array(Edges) #Do not take Edges, seems broken
Edges_X = np.linspace(min(U_eig[:,0]),max(U_eig[:,0]),N_bins + 1)
Edges_Y = np.linspace(min(U_eig[:,1]),max(U_eig[:,1]),N_bins + 1)

X_values = (np.diff(Edges_X) + 2*Edges_X[:-1])/2
Y_values = (np.diff(Edges_Y) + 2*Edges_Y[:-1])/2

KMc_exp = KMc_exp*f_sample #To add time to computations

Centers_X,Centers_Y = np.meshgrid(X_values,Y_values)
Flat_X = Centers_X.flatten()
Flat_Y = Centers_Y.flatten()

### Compute histogram ###

p_hist = np.histogramdd(U_eig[:,:2], bins = bins, density = True)[0]
del U_eig

### Build KM coeffs // Build Sympy ###

KMc = fr.KM_list()

for i in range(len(powers)):
    fun_expr = fr.build_sympy_poly(Coeff_x[i],Coeff_y[i],cross_terms=cross_terms[i])
    coefficient = fr.KM_coefficient(powers[i], fun_expr,KMc_exp[i,:,:].flatten())
    KMc.add_KM_coeff(powers[i], coefficient)

#del fun_expr,coefficient

### Compute values simpy ###
KMc_reg = fr.KM_list(KM_copy=KMc)
KMc_reg.convert_to_langevin_2D_sym()

KMc.compute_sym_values([Centers_X,Centers_Y])
KMc_reg.compute_sym_values([Centers_X,Centers_Y])

### Initialise Xi ###
c_terms = np.array(cross_terms) - 1
N = np.sum(np.array(Coeff_x) + np.array(Coeff_y) + 2)
if np.any(c_terms):
    values = np.array(Coeff_x)
    N += np.sum(values[c_terms]*(values[c_terms] - 1)//2)
Xi0 = np.zeros(N)
size_tot = 0

for k in range(len(powers)):
    fit_KM = KMc.get_fun_value(k)
    Flat_KM = KMc.get_exp_value(k)
    mask = np.nonzero(np.isfinite(Flat_KM))
    Xi0[size_tot:size_tot + len(fit_KM)] = lstsq(np.squeeze(fit_KM[:,mask]).T, Flat_KM[mask], rcond=None)[0]

    size_tot += len(fit_KM)


print('Value of Xi0 :')
print(Xi0)

### COMPUTE WEIGHTS ###

### OPTIMISATION ###
#Compute empirical PDF
W = np.ones((len(powers),N_bins*N_bins))
#Initialise adjoint solver
afp = fpsolve.AdjFP(np.array((X_values,Y_values)),ndim=2,solve='diff',dt=dt)

#Initialise forward steady-state solver
dx = np.array((Edges_X[1] - Edges_X[0],Edges_Y[1] - Edges_Y[0]))
fp = fpsolve.SteadyFP(np.array((N_bins,N_bins)), dx)

#Optimisation parameters
params = {"W": W, "KMc": KMc, "Xi0": Xi0,"N": N_bins, "track" : 0,
          "kl_reg": 10, "p_hist" : p_hist, "Nbr_iter_max" : 2e4,
          "fp": fp, "afp": afp, "tau": stride*delta_t, "radial": False,
          "print_cost": True}

opt_fun_local = lambda params: utils.AFP_opt(utils.cost_reg, params)
Xi, V = utils.SSR_loop(opt_fun_local, params)
