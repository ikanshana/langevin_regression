from kramersmoyal import km, kernels
import netCDF4 as nc
import functions_regression as fr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.io as sio
from numpy.linalg import lstsq
import sympy as sym
import time as t
import copy
import multiprocessing as mp

# path for other librairies. To keep if librairies ar enot in the same folder
sys.path.append(os.path.abspath(
    "/home/administrateur/Documents/lmfl_data/Programmes/python_codes/langevin-regression"))

import fpsolve_reg as fpsolve
import utils_reg as utils

if __name__ == '__main__':  # NECESSARY ! -> Multiprocessing/Multithreading
    ### Parameters ###
    Nbr_proc = 7
    mp.set_start_method("fork")

    type_file = "1.25"

    # Exp data load
    path_file = "/home/administrateur/Documents/lmfl_data/G_H_" + type_file + "_POD_all/"
    name_file_data = "G_H_" + type_file + "_large_POD_all.nc"

    # Python log file
    path_save = "/home/administrateur/Documents/lmfl_data/Regression_results/"
    main_file = open(path_save + "Results_each.txt", 'w')

    # Checkpoint nc file
    save_load = "/home/administrateur/Documents/lmfl_data/Regression_results/save_step.nc"

    N_bins = 22  # KM bins
    stride = 1  # How many snapshots to skip

    f_sample = 4  # Frequency of experimental data (Hz)
    bw = 1e-6  # Band width for kernels

    powers = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [0, 2]])
    Coeff_x = [3, 3, 2, 2, 2]  # Degrees of polynomes
    Coeff_y = [3, 3, 2, 2, 2]
    cross_terms = [True, True, False, False, False]

    name_file_results = "Results_Nbins_" + str(N_bins)

    delta_t = 1/f_sample

    type_solver = 'diff'  # Solver type for KM corrections
    type_multi = 'diff'  # Paralelisation method to use

    checkpoint_load = False

    ### Read Exp Data ###
    file_nc = nc.Dataset(path_file + name_file_data, 'r')

    U_eig = file_nc['eig_vec'][:]

    ### Compute K-M coeffs ###
    bins = np.array([N_bins, N_bins])

    KMc_exp, Edges = km(U_eig[:, :2], bw=bw, bins=bins, powers=powers)
    filtre = KMc_exp == 0.  # Change 0 values into Nan values for conveniances
    KMc_exp[filtre] = np.NaN

    ### Compute Edges ###

    # Edges = np.array(Edges) #Do not take Edges, seems broken
    Edges_X = np.linspace(min(U_eig[:, 0]), max(U_eig[:, 0]), N_bins + 1)
    Edges_Y = np.linspace(min(U_eig[:, 1]), max(U_eig[:, 1]), N_bins + 1)

    X_values = (np.diff(Edges_X) + 2*Edges_X[:-1])/2
    Y_values = (np.diff(Edges_Y) + 2*Edges_Y[:-1])/2

    KMc_exp = KMc_exp*f_sample  # To add time to computations

    Centers_X, Centers_Y = np.meshgrid(X_values, Y_values)
    Flat_X = Centers_X.flatten()
    Flat_Y = Centers_Y.flatten()

    ### Compute histogram ###

    p_hist = np.histogramdd(U_eig[:, :2], bins=bins, density=True)[0]
    del U_eig  # Free memory

    ### Build KM coeffs // Build Sympy ###

    KMc = fr.KM_list()

    for i in range(len(powers)):
        fun_expr = fr.build_sympy_poly(
            Coeff_x[i], Coeff_y[i], cross_terms=cross_terms[i])
        coefficient = fr.KM_coefficient(
            powers[i], fun_expr, KMc_exp[i, :, :].flatten())
        KMc.add_KM_coeff(powers[i], coefficient)

    ### Compute values simpy ###
    KMc_reg = fr.KM_list(KM_copy=KMc)
    KMc_reg.convert_to_langevin_2D_sym()

    KMc.compute_sym_values([Centers_X, Centers_Y])
    KMc_reg.compute_sym_values([Centers_X, Centers_Y])

    ### Initialise Xi ###
    # Count polynome regular terms
    N = np.sum(np.array(Coeff_x) + np.array(Coeff_y) + 2)

    if np.any(cross_terms):  # Count polynome cross terms
        values = np.array(Coeff_x)
        N += np.sum(values[cross_terms]*(values[cross_terms] - 1)//2)

    Xi0 = np.zeros(N)
    size_tot = 0

    for k in range(len(powers)):  # Natural guess with lstsq
        fit_KM = KMc_reg.get_fun_value(k)
        Flat_KM = KMc_reg.get_exp_value(k)
        mask = np.nonzero(np.isfinite(Flat_KM))
        Xi0[size_tot:size_tot + len(fit_KM)] = lstsq(
            np.squeeze(fit_KM[:, mask]).T, Flat_KM[mask], rcond=None)[0]

        size_tot += len(fit_KM)

    print('Value of Xi0 :')
    print(Xi0)

    ### COMPUTE WEIGHTS ###
    W = np.ones((len(powers), N_bins*N_bins))
    ### OPTIMISATION ###
    # Initialise adjoint solver
    afp = fpsolve.AdjFP(np.array((X_values, Y_values)),
                        ndim=2, solve=type_solver)

    # Initialise forward steady-state solver
    dx = np.array((Edges_X[1] - Edges_X[0], Edges_Y[1] - Edges_Y[0]))
    fp = fpsolve.SteadyFP(np.array((N_bins, N_bins)), dx)

    # Optimisation parameters
    params = {"W": W, "KMc": KMc, "Xi0": Xi0, "N": N_bins, "track": 0,
              "kl_reg": 10, "p_hist": p_hist, "Nbr_iter_max": 5e4,
              "fp": fp, "afp": afp, "tau": stride*delta_t, "file_results": main_file,
              "Nbr_proc": Nbr_proc, "print_cost": True, "mod_multi": type_multi,
              "checkpoint_file": save_load, "checkpoint_load": checkpoint_load}

    def opt_fun_local(params): return utils.AFP_opt(utils.cost_reg, params)

    if params["mod_multi"] == "step":
        Xi, V = utils.SSR_loop_multi(opt_fun_local, params)
    if params["mod_multi"] == "diff":
        Xi, V = utils.SSR_loop(opt_fun_local, params)
    else:
        print("Error : Only two multi-processing types supported")

    np.savetxt(path_save + name_file_results + "_Xi.csv", Xi)
    np.savetxt(path_save + name_file_results + "_V.csv", V)
    main_file.close()
