"""
classy.py

Drop-in stand-in for the real `classy` (CLASS's Python wrapper), backed
by CosmoPower instead of an actual Boltzmann solve. MontePython does:

    from classy import Class
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    cl = cosmo.lensed_cl(lmax)

This module provides that same `Class` name and interface. As long as
THIS file's directory is found on sys.path BEFORE any real classy
installation, `import classy` resolves here instead -- MontePython
requires no source changes.

FIXED (non-sampled) CONFIGURATION
----------------------------------
l_switch (the ell bin edges) and the path to the pretrained CosmoPower
network are configuration, not MCMC parameters -- set via environment
variables so you don't have to edit this file to change them:

    export COSMOPOWER_TT_PATH=/path/to/cmb_TT_NN
    export L_SWITCH=800,1600,2508

MCMC parameters (varied by MontePython, declared in your .param file,
passed through cosmo.set() exactly as CLASS parameters would be):

    omega_b, h, tau_reio, n_s, ln10^{10}A_s   (shared across all bins)
    Om_bin0, Om_bin1, ..., Om_bin{n_bins-1}   (one omega_m per bin)

CAVEATS -- READ BEFORE TRUSTING ANY OUTPUT OF A RUN USING THIS SHIM
--------------------------------------------------------------------
1. Derived parameters (Omega_m, H0, sigma8, age, rs_drag, ...) have no
   single well-defined value in a model where omega_m varies by ell
   bin. get_current_derived_parameters() below uses Om_bin0 as an
   arbitrary reference -- this is a placeholder convention, not
   physics. If any likelihood or prior in your MontePython run
   actually depends on one of these (e.g. a BAO likelihood using
   Omega_m/H(z)), that dependence is NOT meaningful for this model and
   you should not include such likelihoods until this is resolved.

2. Lensed vs unlensed spectra: raw_cl() and lensed_cl() both currently
   return the SAME patched array from the "cmb_TT_NN" network. Verify
   from the CosmoPower repo/paper which one that specific pretrained
   network actually emulates, and fix this file (and/or swap in a
   separate lensing-potential "PP" network) accordingly before using
   any likelihood that distinguishes the two (e.g. a lensing
   likelihood).

3. TE, EE, BB, PP are returned as zero arrays -- this shim only wires
   up TT. If your likelihood uses TE/EE (as Planck 2018 does), you
   need to load the corresponding CosmoPower TE/EE networks and patch
   them the same way TT is patched below.

4. Hubble(z) raises NotImplementedError. Background evolution isn't
   well-defined for this model; any likelihood/prior calling it will
   fail loudly rather than silently returning something meaningless.
"""

import os
import numpy as np
from cosmopower import cosmopower_NN

# ---------------------------------------------------------------
# Fixed, non-sampled configuration
# ---------------------------------------------------------------
CP_TT_PATH = os.environ.get(
    "COSMOPOWER_TT_PATH",
    "/Users/bellamarco01/uni/omega_m/cosmopower/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN",
)
L_SWITCH = np.array(
    [float(x) for x in os.environ.get("L_SWITCH", "800,1600,2508").split(",")]
)
N_BINS = len(L_SWITCH)
CDM_PARAM_NAME = os.environ.get("COSMOPOWER_CDM_PARAM_NAME", "omega_cdm")

_cp_tt = cosmopower_NN(restore=True, restore_filename=CP_TT_PATH)
_ELL = np.asarray(_cp_tt.modes)


def _bin_masks():
    edges = np.concatenate(([_ELL.min()], L_SWITCH))
    masks = []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            m = (_ELL >= lo) & (_ELL <= hi)
        else:
            m = (_ELL > lo) & (_ELL <= hi)
        masks.append(m)
    n_assigned = sum(m.sum() for m in masks)
    if n_assigned != len(_ELL):
        raise ValueError(
            f"L_SWITCH={L_SWITCH} does not partition the emulator's ell "
            f"grid ({n_assigned} of {len(_ELL)} assigned). Check L_SWITCH."
        )
    return masks


_MASKS = _bin_masks()


class CosmoSevereError(Exception):
    """Mirrors classy's exception name so likelihood code that catches
    it specifically still works."""
    pass


class CosmoComputationError(Exception):
    """Mirrors classy's exception name."""
    pass


class Class:
    """
    Minimal duck-typed stand-in for classy.Class(), enough to drive a
    TT-only likelihood on the binned-omega_m CosmoPower model. Extend
    the TE/EE/derived-parameter sections (see module docstring) before
    using anything beyond TT.
    """

    def __init__(self):
        self._params = {}
        self._cl_patched_linear = None

    # ---- classy-compatible interface -------------------------------
    def set(self, params):
        self._params.update(params)

    def compute(self, level=None):
        required = ["omega_b", "h", "tau_reio", "n_s", "ln10^{10}A_s"]
        missing = [p for p in required if p not in self._params]
        missing += [f"Om_bin{i}" for i in range(N_BINS) if f"Om_bin{i}" not in self._params]
        if missing:
            raise CosmoSevereError(
                f"missing required parameters in classy shim: {missing}"
            )

        omega_b = self._params["omega_b"]
        base_params = {
            "omega_b": omega_b,
            "h": self._params["h"],
            "tau_reio": self._params["tau_reio"],
            "n_s": self._params["n_s"],
            "ln10^{10}A_s": self._params["ln10^{10}A_s"],
        }
        omega_m_bins = np.array(
            [self._params[f"Om_bin{i}"] for i in range(N_BINS)]
        )

        cl_patched = np.zeros_like(_ELL, dtype=float)
        for mask, om in zip(_MASKS, omega_m_bins):
            p = dict(base_params)
            p[CDM_PARAM_NAME] = om - omega_b  # omega_m -> omega_cdm
            cp_params = {k: np.atleast_1d(v) for k, v in p.items()}
            log_cl_full = _cp_tt.predictions_np(cp_params)[0]
            cl_patched[mask] = 10.0 ** log_cl_full[mask]  # un-log

        self._cl_patched_linear = cl_patched

    def _cl_dict(self, lmax):
        ell = _ELL
        cl = self._cl_patched_linear
        if cl is None:
            raise CosmoSevereError("compute() must be called before requesting Cl's")
        if lmax is not None:
            keep = ell <= lmax
            ell = ell[keep]
            cl = cl[keep]
        zeros = np.zeros_like(cl)
        return {"ell": ell, "tt": cl, "te": zeros, "ee": zeros, "bb": zeros, "pp": zeros}

    def raw_cl(self, lmax=None):
        return self._cl_dict(lmax)

    def lensed_cl(self, lmax=None):
        # see CAVEAT 2 in the module docstring
        return self._cl_dict(lmax)

    def get_current_derived_parameters(self, names):
        # see CAVEAT 1 in the module docstring
        h = self._params["h"]
        omega_m_ref = self._params["Om_bin0"]
        out = {}
        for name in names:
            if name == "H0":
                out[name] = 100.0 * h
            elif name == "Omega_m":
                out[name] = omega_m_ref / h**2
            else:
                raise NotImplementedError(
                    f"derived parameter '{name}' is not implemented in "
                    f"the CosmoPower classy shim (see classy.py CAVEAT 1)"
                )
        return out

    def T_cmb(self):
        return 2.7255

    def Omega_m(self):
        return self._params["Om_bin0"] / self._params["h"] ** 2

    def Hubble(self, z=0):
        raise NotImplementedError(
            "Hubble(z) has no single well-defined value for this "
            "binned-omega_m model (see classy.py CAVEAT 4)"
        )

    def struct_cleanup(self):
        pass

    def empty(self):
        self._params = {}
        self._cl_patched_linear = None
