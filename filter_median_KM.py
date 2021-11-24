import numpy as np
import copy
import time as t
import sys
import os
import scipy.io as sio
import matplotlib.pyplot as plt

import netCDF4 as nc
from kramersmoyal import km,kernels
from openpiv import validation, filters

### Parameters ###
type_file = "1.25"

path_file = "/home/administrateur/Documents/lmfl_data/G_H_" + type_file + "_POD_all/"
name_file = "G_H_" + type_file + "_large_POD_all.nc"

N_bins = 22 #KM bins
stride = 1 #How many snapshots to skip

f_sample = 4
bw = 1e-6 #Band width for kernels

powers = np.array([[1,0], [0,1], [1,1], [2,0], [0,2]])
delta_t = 0.25

### Read Exp Data ###
file_nc = nc.Dataset(path_file + name_file,'r')

U_eig = file_nc['eig_vec'][:]

### Compute K-M coeffs + Edges ###
bins = np.array([N_bins, N_bins])

KMc_exp, Edges = km(U_eig[:,:2],bw = bw, bins = bins, powers = powers)
#filtre = KMc_exp == 0.
#KMc_exp[filtre] = np.NaN

Edges = np.array(Edges) #Do not take Edges, seems broken

Edges_X = np.linspace(min(U_eig[:,0]),max(U_eig[:,0]),N_bins + 1)
Edges_Y = np.linspace(min(U_eig[:,1]),max(U_eig[:,1]),N_bins + 1)

del U_eig

X_values = (np.diff(Edges_X) + 2*Edges_X[:-1])/2
Y_values = (np.diff(Edges_Y) + 2*Edges_Y[:-1])/2

KMc_exp = KMc_exp*f_sample #To add time to computations

Centers_X,Centers_Y = np.meshgrid(X_values,Y_values)
Flat_X = Centers_X.flatten()
Flat_Y = Centers_Y.flatten()

### Select purious data ###
f_x = KMc_exp[0,:,:]
f_y = KMc_exp[1,:,:]

f_x_thresh = 0.03
f_y_thresh = 0.03

f_x2 = copy.deepcopy(f_x)
f_y2 = copy.deepcopy(f_y)

f_x_filt,f_y_filt,mask = validation.local_median_val(f_x2,f_y2,f_x_thresh,f_y_thresh)
filtre = f_x2 == 0.
f_x_filt[filtre] = np.NaN
f_y_filt[filtre] = np.NaN



plt.figure(figsize=[1600//96,900//96])
plt.quiver(Centers_X,Centers_Y,f_x.T,f_y.T,scale=1)
plt.quiver(Centers_X[mask],Centers_Y[mask],f_x.T[mask],f_y.T[mask],color='r',scale=1)
#plt.quiver(Centers_X[mask2],Centers_Y[mask2],f_x.T[mask2],f_y.T[mask2],color='g')

f_x_rep,f_y_rep = filters.replace_outliers(f_x_filt,f_y_filt,
                                            method ='localmean',
                                            max_iter=1,
                                            kernel_size=3)
f_x_rep[filtre] = np.NaN
f_y_rep[filtre] = np.NaN

plt.figure(figsize=[1600//96,900//96])
plt.quiver(Centers_X,Centers_Y,f_x_rep.T,f_y_rep.T,scale=1)
#plt.quiver(Centers_X,Centers_Y,f_x_filt.T,f_y_filt.T)
