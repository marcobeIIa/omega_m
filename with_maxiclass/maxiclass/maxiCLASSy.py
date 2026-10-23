import sys
import classy
from classy import Class
print(classy.__file__)

import os
import copy
import yaml
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from matplotlib import rc
from scipy.interpolate import interp1d

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
#matplotlib.rc('font', **font)
matplotlib.rcParams['legend.fontsize']='medium'
plt.rcParams["figure.figsize"] = [8.0,6.0]

params_mEDE = {
    # "compute_sigma8": "yes",
    "write background": 'y',
    "root": "output/maxiClassy",
    "output": "tCl,lCl,mPk",
    "omega_b": 0.02251,
    "omega_cdm": 0.1320,
    "H0": 72.81,
    "tau_reio": 0.068,
    "A_s": 2.191e-9,
    "n_s": 0.9860,

    "N_ur": 2.0328,
    "N_ncdm": 1,
    "deg_ncdm": 1, 
    "m_ncdm": 0.06,
    "T_ncdm": 0.7161,

    "N_mscf": 3,
    "n_axion_mscf": '3,3,3',
    #"f_axion_mscf": 0.1,
    #"m_mscf": 1e4,
    "theta_ini_mscf": '2.7,2.7,2.7',
    "theta_prime_ini_mscf": '0.0,0.0,0.0',
    "do_shooting_mscf": 'y',
    "do_shooting": 'y',
    "tol_shooting_deltax":0.01,
    "tol_shooting_deltaF":0.01,
    "background_Nloga": 100000,
    # "alpha_squared_mscf": -30.,
    # "power_of_mu_mscf": -30.,
    "log10_maxion_ac": '-3,-3.5,-4',
    "fraction_maxion_ac": '0.1,0.1,0.1',

    "input_verbose": 12,
    "background_verbose": 12,
    "thermodynamics_verbose": 1,
    "perturbations_verbose": 1,
    "transfer_verbose": 1,
    "primordial_verbose": 1,
    # "spectra_verbose": 1,
    # "nonlinear_verbose": 1,
    "lensing_verbose": 1,
    "output_verbose": 1,
    "l_max_scalars":5000,
}
EDEm = Class()
EDEm.set(params_mEDE)
EDEm.compute()
#bg_m = EDEm.get_background()
