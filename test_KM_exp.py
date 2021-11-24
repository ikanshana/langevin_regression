import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.io as sio

from scipy.stats import kurtosis

sys.path.append(os.path.abspath("/home/administrateur/Documents/lmfl_data/Programmes/python_codes/langevin-regression"))

import utils

### Parameters ###
N_start = 1
N_end = 10
dimt = 2000
N_run = N_end - N_start + 1
N_mean = 4
dt = 1/4
Nr = 24 #Number of histogram bins

type_file = '2.4'
type_method = 'tip'

### LOAD DATA ###
path = "/home/administrateur/Documents/lmfl_data/Time_distributions_" + type_method + "_" + type_file + "/"
path_save = "/home/administrateur/Documents/lmfl_data/Figures/Essais_modele/"

file_save = "autocorr_Markov_" + type_file + "_" + type_method

Ym_tot = np.zeros([N_run*dimt,1])

for run in range(N_start,N_end + 1):
	file_data = "time_distribution_run" + f'{run:02d}' + ".mat"
	
	data = sio.loadmat(path + file_data)
	
	Ym_tot[(run - N_start)*dimt:(run - N_start + 1)*dimt] = data['Ym']

### CORRELATION FUNCTION ###

R = utils.autocorr_func_1d(Ym_tot[:,0])
tau = dt*np.arange(0,len(R))

plt.figure(figsize=(12,10))
plt.subplot(3,1,1)
plt.plot(tau,R,'k')
plt.ylabel(r'$C(\tau)$',fontsize=18)
plt.xlabel(r'Time lag $\tau$',fontsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.gca().set_xscale('log')

plt.grid()

### MARKOV PROPERTY ###

lag = np.round( np.logspace(0, np.log10(20000), 50) ).astype(int)
kl_div = np.array([utils.markov_test(Ym_tot[:,0], delta, N=32) for delta in lag])

plt.subplot(3,1,2)
plt.gca().set_xscale('log')
plt.plot(dt*lag,kl_div,'k.')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.ylabel(r'$\mathcal{D}_{KL}(\tau)$',fontsize=18)
plt.xlabel(r'Sampling time $\tau$',fontsize=18)

plt.grid()

### KURTOSIS ###
r_edges = np.linspace(min(Ym_tot), max(Ym_tot), Nr+1)
lag = np.arange(1, 1000)

kurt = np.zeros(lag.shape)
for j in range(len(lag)):
    tau = dt*lag[j]
    dX = (Ym_tot[lag[j]:] - Ym_tot[:-lag[j]])/tau  # Step (finite-difference derivative estimate)

    for i in range(Nr):
        # Find where signal falls into this bin
        mask = np.nonzero( (Ym_tot[:-lag[j]] > r_edges[i]) * (Ym_tot[:-lag[j]] < r_edges[i+1]))[0]
        value = kurtosis(dX[mask], fisher=False,nan_policy='omit')
        
        if np.isnan(value) == 0:
        	kurt[j] += kurtosis(dX[mask], fisher=False,nan_policy='omit')

kurt = kurt/Nr # Average over domain

plt.subplot(3,1,3)
plt.plot(dt*lag, kurt, 'k')
plt.gca().set_xscale('log')
plt.grid()
plt.xlabel(r'Time lag $\tau$')
plt.ylabel('Kurtosis')

plt.show()

### SAVE DATA ###
plt.savefig(path_save + file_save + '.svg')
plt.savefig(path_save + file_save + '.png')
s
