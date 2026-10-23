##
## MADRINI.py

## stands for

## M(ass) padrini.py

## this version creates an array of masses and creates an ini file

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
#        f_axion_mscf = params.get("f_axion_mscf", 0.01)
        n_axion_mscf = params.get("n_axion_mscf", 1)

        # mass # Mass range in log10 (default: 1e-5 to 1e-3)
        log10_m_min, log10_m_max = params.get("log10_m_range", (5, 3))
        f_min, f_max = params.get("f_axion_range",(.05,2))

        # Generate N linear masses (log-spaced)
        m_vals = np.logspace(log10_m_min, log10_m_max, N)
        f_vals = np.linspace(f_min, f_max, N)


        # Write N_mscf
        lines.append(f"N_mscf = {N}")

        # Expand repeated parameters
        lines.append("theta_ini_mscf=" + ",".join([str(theta_ini)] * N))
        lines.append("theta_prime_ini_mscf=" + ",".join([str(theta_prime_ini)] * N))
        #lines.append("f_axion_mscf = " + " , ".join([str(f_axion_mscf)] * N))
        lines.append("n_axion_mscf=" + ",".join([str(n_axion_mscf)] * N))
        lines.append("m_mscf=" + ",".join([f"{val:.2e}" for val in m_vals]))
        lines.append("f_axion_mscf=" + ",".join([f"{val:.2e}" for val in f_vals]))

    # Handle all other parameters (skip N_mscf ones already written)
    #skip_keys = {"header", "N_mscf", "theta_ini_mscf", "theta_prime_ini_mscf", 
    #             "log10_m_range", "n_axion_mscf", "f_axion_mscf"}
    skip_keys = {"header", "N_mscf", "theta_ini_mscf", "theta_prime_ini_mscf", 
                 "log10_m_range", "n_axion_mscf"}
    
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
#    "output": "tCl,lCl,mPk",
    "P_k_max_h/Mpc": 1,
    "write background": "yes",
    "write parameters": "yes",
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
#    "do_shooting": 'n',
#    "do_shooting_mscf": 'n',
    "input_verbose": 12,
    "background_verbose": 10,
    "thermodynamics_verbose": 12,
    "output_verbose":12,

    # Special block
    "N_mscf":3,
    "theta_ini_mscf": 2.6,
    "theta_prime_ini_mscf": 0.0,
#    "f_axion_mscf": 0.1,
    "f_axion_range": (0.1,0.1),
    "n_axion_mscf": 3,
    "log10_m_range": (0,8),
}

write_class_ini(params, "madr.ini")

# Write your madr.ini somewhere above in the code...
ini_file = "madr.ini"

# Run CLASS with that ini
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
