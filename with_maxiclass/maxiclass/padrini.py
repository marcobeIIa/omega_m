##
## PADRINI.py

## Parameter guess for
## Axion 
## Dark energy
## Redshift range 
## Ini (.ini) creator

## .py stands for python


import numpy as np
import subprocess

def write_class_ini(params, filename):
    """
    Write CLASS .ini parameter file.
    
    Special handling for N_mscf: generates multiple entries for 
    theta_ini, theta_prime_ini, fraction_maxion_ac, n_axion_mscf,
    and log10_maxion_ac over a given eedshift range.

    Parameters:
    params (dict): Dictionary of parameters.
                   filename (str): Output filename (.ini).
                                   """

    lines = []

    # Handle header if provided
    if "header" in params:
        lines.append(params["header"])

    # Handle N_mscf block
    if "N_mscf" in params:
        N = params["N_mscf"]

    # Base values
    theta_ini = params.get("theta_ini_mscf", 0.1)
    theta_prime_ini = params.get("theta_prime_ini_mscf", 0.0)
    fraction_maxion_ac = params.get("fraction_maxion_ac", 0.01)
    n_axion_mscf = params.get("n_axion_mscf", 1)

                                   # Redshift range
    log10_ac_min, log10_ac_max = params.get("log10_ac_range", (-3,-5))
    log10_vals = np.linspace(log10_ac_min, log10_ac_max, N)

    # Write N_mscf
    lines.append(f"N_mscf = {N}")

    # Expand repeated parameters
    lines.append("theta_ini_mscf = " + " , ".join([str(theta_ini)] * N))
    lines.append("theta_prime_ini_mscf = " + " , ".join([str(theta_prime_ini)] * N))
    lines.append("fraction_maxion_ac = " + " , ".join([str(fraction_maxion_ac)] * N))
    lines.append("n_axion_mscf = " + " , ".join([str(n_axion_mscf)] * N))
    lines.append("log10_maxion_ac = " + " , ".join([f"{val:.2f}" for val in log10_vals]))

    # Handle all other parameters (skip N_mscf ones already written)
    skip_keys = {"header", "N_mscf", "theta_ini_mscf", "theta_prime_ini_mscf", 
                 "fraction_maxion_ac", "n_axion_mscf", "log10_ac_range"}
    
    for key, val in params.items():
        if key not in skip_keys:
            lines.append(f"{key} = {val}")

    # Save to file
    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"INI file written to {filename}")

params = {
    "header": "*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*\n"
              "*  CLASS input parameter file (auto-generated)       *\n"
              "*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*",
    "root": "output/mb_example_axiclass",
    "output": "tCl,lCl,mPk",
    "P_k_max_h/Mpc": 1,
    "write background": "no",
    "write parameters": "no",
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
    "T_ncdm": 0.71611,
    "do_shooting": 'y',
    "do_shooting_mscf": 'y',
    "input_verbose": 11,
    "background_verbose": 10,
    "thermodynamics_verbose": 4,
    "output_verbose":5,
    "tol_shooting_deltax":0.01,
    "tol_shooting_deltaF":0.01,
    "background_Nloga": 100000,
    # Special block
    "N_mscf": 10,
    "theta_ini_mscf": 2.6,
    "theta_prime_ini_mscf": 0.0,
    "fraction_maxion_ac": 0.05,
    "n_axion_mscf": 3,
    "log10_ac_range": (-3,-4.5),
}

write_class_ini(params, "padr.ini")

# Write your padr.ini somewhere above in the code...
ini_file = "padr.ini"

# Run CLASS with that ini
run_class = 0 #this parameter decides if we are just computing the ini array or actually running class
if run_class:
    try:
        result = subprocess.run(
            ["./class", ini_file],
            check=True,
       #     capture_output=True,
       #    text=True
        )
        print("CLASS finished successfully.")
        print("stdout:\n", result.stdout)
        print("stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error while running CLASS:")
        print(e.stderr)
