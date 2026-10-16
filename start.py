import numpy as np
from cosmopower import cosmopower_NN
from model import BinnedOmegaMEmulator
import matplotlib.pyplot as plt

cp_nn = cosmopower_NN(
    restore=True,
    restore_filename="/Users/bellamarco01/uni/omega_m/with_cosmopower/cosmopower/cosmopower/trained_models/CP_paper/CMB"
)

print(cp_nn.parameters)
# -> ['omega_b', 'omega_cdm', 'h', 'tau_reio', 'n_s', 'ln10^{10}A_s']

base_params = {
    "omega_b": 0.0224,
    "h": 0.674,
    "tau_reio": 0.054,
    "n_s": 0.965,
    "ln10^{10}A_s": 3.044,
}
# NOTE: no omega_cdm and no omega_m in base_params -- omega_cdm is
# derived per-bin inside the class from omega_m_bins and omega_b.

emu = BinnedOmegaMEmulator(cp_emulator=cp_nn, base_params=base_params)

l_switch = [800, 1600, 2508]
omega_m_bins = [0.14, 0.15, 0.16]
ell, cl_patched = emu.predict(l_switch, omega_m_bins)
plt.plot(ell, cl_patched * ell * (ell + 1))
plt.show()
