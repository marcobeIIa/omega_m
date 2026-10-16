"""
classy.py -- CosmoPower-backed stand-in for CLASS's Python wrapper
==================================================================

MontePython does `from classy import Class` and then drives the object with
set() / compute() / lensed_cl() / struct_cleanup() / empty().  This module
provides a `Class` with that same interface, but the spectra come from the
pretrained CosmoPower CMB emulators (Spurio Mancini et al. 2022,
arXiv:2106.03846), evaluated piecewise in ell so that omega_m can take a
different value in each multipole bin.

What the emulators are (checked against real CLASS v3.4, Planck-2018 LCDM):
    cmb_TT_NN          cosmopower_NN        -> log10 C_l^TT   (lensed)
    cmb_EE_NN          cosmopower_NN        -> log10 C_l^EE   (lensed)
    cmb_TE_PCAplusNN   cosmopower_PCAplusNN ->       C_l^TE   (lensed; not
                                                    logged, TE changes sign)
    cmb_PP_PCAplusNN   cosmopower_PCAplusNN -> log10 C_l^phiphi
    all dimensionless, every multipole 2 <= l <= 2508,
    inputs: omega_b, omega_cdm, h, tau_reio, n_s, ln10^{10}A_s.
Median agreement with CLASS lensed_cl: TT 0.03%, EE 0.2%, TE 0.2%, PP 0.3%.

Faithful to real classy (so MontePython's Planck / mock-CMB likelihoods work):
  * lensed_cl(lmax=-1) returns a dict of arrays indexed by ell from 0
    (entries at l=0,1 are zero), keys 'ell','tt','ee','te','pp','tp',
    dimensionless -- MontePython multiplies by (T_cmb*1e6)^2 itself.
  * lmax=-1 means "up to l_max_scalars" (or 2508 if not set);
    lmax beyond what was computed raises CosmoSevereError.
  * set(dict) / set(**kw), compute(level), struct_cleanup(), empty(),
    .state, T_cmb(), get_current_derived_parameters([...]).
  * Exceptions: CosmoComputationError for a parameter point the code can't
    evaluate (MontePython rejects the step), CosmoSevereError for
    configuration errors (MontePython stops).

NOT like real classy -- deliberate, read before trusting a run:
  1. The binning itself.  Required MCMC params: omega_b, h, tau_reio, n_s,
     ln10^{10}A_s (or A_s), omega_m0..omega_m{N-1}, l_switch1..l_switch{N}.
     Bin 0 is [2, l_switch1], bin i is (l_switch{i}, l_switch{i+1}];
     l_switch{N} must be 2508.  In bin i every spectrum (TT, TE, EE and PP)
     is taken from the emulator evaluated at omega_cdm = omega_m{i} - omega_b.
     Applying the same ell-bins to PP is a modelling choice: the lensing
     potential's L is not the CMB's l.  For a single bin (N=1,
     l_switch1=2508) the model is exactly LCDM.
  2. raw_cl() (UNlensed spectra) raises: no unlensed emulator is loaded.
     MontePython's Planck and mock-CMB likelihoods use lensed_cl().
  3. The CP_paper emulators provide neither lensed BB nor the T-phi cross.
     - 'bb' is left OUT of the dict, so anything that needs it fails loudly
       with a KeyError (e.g. Planck_lowl_EEBB -- use Planck_lowl_EE -- or
       Bmodes = True in mock-CMB likelihoods).
     - 'tp' is returned as ZEROS, because MontePython's mock-CMB likelihood
       writes it into the fiducial file even when it ignores it.  Always
       keep neglect_TD = True there (the default), which multiplies it by 0.
       The real Planck likelihoods never read 'tp'.
  4. No background: Hubble(z), distances, sigma8, rs_drag, theta_s... raise.
     get_current_derived_parameters() only knows quantities that are plain
     functions of the inputs; 'Omega_m'/'omega_m' refer to bin 0 (a
     placeholder convention, not physics).  Don't combine with BAO/SN
     likelihoods.
  5. Points outside the emulator training box raise CosmoComputationError
     (the step is rejected), per bin:
        omega_b [0.005,0.04], omega_cdm [0.001,0.99], h [0.2,1.0],
        tau_reio [0.01,0.8], n_s [0.7,1.3], ln10^{10}A_s [1.61,5].

Configuration (environment variables, optional):
    COSMOPOWER_CMB_DIR   directory holding the four CP_paper/CMB .pkl files
                         (default: ../../cosmopower/cosmopower/trained_models/
                         CP_paper/CMB relative to this file)
    COSMOPOWER_CDM_PARAM_NAME   emulator's CDM input name (default omega_cdm)
"""

import os
import numpy as np
from cosmopower import cosmopower_NN, cosmopower_PCAplusNN

# ---------------------------------------------------------------------------
# Emulators (loaded once, at import)
# ---------------------------------------------------------------------------
# Default: the cosmopower repo sitting next to this class/ folder
# (with_cosmopower/class/python/classy.py -> with_cosmopower/cosmopower/...).
CP_CMB_DIR = os.environ.get(
    "COSMOPOWER_CMB_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "cosmopower", "cosmopower",
                                  "trained_models", "CP_paper", "CMB")),
)
CDM_PARAM_NAME = os.environ.get("COSMOPOWER_CDM_PARAM_NAME", "omega_cdm")

# name -> (network class, file stem, output is log10?)
_NETWORKS = {
    "tt": (cosmopower_NN, "cmb_TT_NN", True),
    "ee": (cosmopower_NN, "cmb_EE_NN", True),
    "te": (cosmopower_PCAplusNN, "cmb_TE_PCAplusNN", False),
    "pp": (cosmopower_PCAplusNN, "cmb_PP_PCAplusNN", True),
}
_EMU = {
    k: (cls(restore=True, restore_filename=os.path.join(CP_CMB_DIR, stem)), is_log)
    for k, (cls, stem, is_log) in _NETWORKS.items()
}

_ELL = np.asarray(_EMU["tt"][0].modes, dtype=int)  # 2..2508
for _k, (_e, _) in _EMU.items():
    if not np.array_equal(np.asarray(_e.modes, dtype=int), _ELL):
        raise ImportError("CosmoPower network '%s' has a different ell grid than TT" % _k)
L_MIN_EMU, L_MAX_EMU = int(_ELL[0]), int(_ELL[-1])
# lensed_cl() maps emulator array index i -> multipole L_MIN_EMU + i. That is only
# valid if the emulator grid is every integer multipole with no gaps; refuse to
# load otherwise (this is exactly the kind of silent ell shift we must never have).
if not np.array_equal(_ELL, np.arange(L_MIN_EMU, L_MAX_EMU + 1)):
    raise ImportError("CosmoPower ell grid is not contiguous: %s ... %s" % (_ELL[:5], _ELL[-5:]))

_TRAINING_RANGE = {
    "omega_b": (0.005, 0.04),
    CDM_PARAM_NAME: (0.001, 0.99),
    "h": (0.2, 1.0),
    "tau_reio": (0.01, 0.8),
    "n_s": (0.7, 1.3),
    "ln10^{10}A_s": (1.61, 5.0),
}

_T_CMB = 2.7255  # K; CLASS default, which the CP_paper training sets used


# ---------------------------------------------------------------------------
# Exceptions (same names and hierarchy as classy)
# ---------------------------------------------------------------------------
class CosmoError(Exception):
    def __init__(self, message=""):
        self.message = message.decode() if isinstance(message, bytes) else message

    def __str__(self):
        return "\n\nError in Class: " + self.message


class CosmoSevereError(CosmoError):
    """Configuration problem -- MontePython stops."""


class CosmoComputationError(CosmoError):
    """This parameter point can't be computed -- MontePython rejects the step."""


# ---------------------------------------------------------------------------
# ell-bin bookkeeping
# ---------------------------------------------------------------------------
_MASK_CACHE = {}


def _bin_masks(l_switch):
    """Bin 0 = [2, l_switch[0]], bin i = (l_switch[i-1], l_switch[i]]."""
    key = tuple(float(x) for x in l_switch)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    if any(b <= a for a, b in zip(key[:-1], key[1:])):
        raise CosmoSevereError("l_switch values must be strictly increasing: %s" % (key,))
    if key[-1] != L_MAX_EMU:
        raise CosmoSevereError(
            "the last l_switch must equal the emulator's l_max = %d (got %g)"
            % (L_MAX_EMU, key[-1]))
    edges = (L_MIN_EMU - 1,) + key  # so bin 0 includes ell = 2
    masks = [(_ELL > lo) & (_ELL <= hi) for lo, hi in zip(edges[:-1], edges[1:])]
    _MASK_CACHE[key] = masks
    return masks


# ---------------------------------------------------------------------------
# The Class stand-in
# ---------------------------------------------------------------------------
class Class(object):

    def __init__(self, default=False):
        self._pars = {}
        self._cl = None             # dict of arrays on _ELL, set by compute()
        self._computed_pars = None  # snapshot of _pars used for _cl
        self._lmax = L_MAX_EMU

    # -- state ------------------------------------------------------------
    @property
    def state(self):
        # MontePython: "has compute() been run (and not cleaned up)?"
        return self._cl is not None

    def pars(self):
        return self._pars

    # -- input ------------------------------------------------------------
    def set(self, *pars, **kars):
        if len(pars) == 1:
            self._pars.update(dict(pars[0]))
        elif len(pars) != 0:
            raise CosmoSevereError("bad call")
        self._pars.update(kars)
        return True

    def empty(self):
        self._pars = {}
        self.struct_cleanup()

    def struct_cleanup(self):
        self._cl = None
        self._computed_pars = None

    # -- parameter parsing --------------------------------------------------
    def _get(self, name):
        try:
            return float(self._pars[name])
        except KeyError:
            raise CosmoSevereError("missing required parameter '%s'" % name)

    def _l_switch(self):
        vals, i = [], 1
        while "l_switch%d" % i in self._pars:
            vals.append(float(self._pars["l_switch%d" % i]))
            i += 1
        if not vals:
            raise CosmoSevereError(
                "no l_switch1, l_switch2, ... parameters given (use a single "
                "l_switch1 = %d for plain LCDM)" % L_MAX_EMU)
        return np.array(vals)

    def _ln10As(self):
        if "ln10^{10}A_s" in self._pars:
            return float(self._pars["ln10^{10}A_s"])
        if "A_s" in self._pars:
            return float(np.log(1e10 * float(self._pars["A_s"])))
        raise CosmoSevereError("missing scalar amplitude: give ln10^{10}A_s or A_s")

    def _requested_lmax(self):
        lmax = int(float(self._pars.get("l_max_scalars", L_MAX_EMU)))
        if lmax > L_MAX_EMU:
            raise CosmoSevereError(
                "l_max_scalars = %d requested, but the CosmoPower emulators stop at "
                "l = %d. Lower l_max in the likelihood's .data file." % (lmax, L_MAX_EMU))
        return lmax

    # -- the computation ----------------------------------------------------
    def compute(self, level=None):
        # Nothing changed since the last compute -> keep the spectra.
        if self._cl is not None and self._computed_pars == self._pars:
            return

        l_switch = self._l_switch()
        n_bins = len(l_switch)
        masks = _bin_masks(l_switch)
        lmax = self._requested_lmax()

        omega_b = self._get("omega_b")
        omega_m = np.array([self._get("omega_m%d" % i) for i in range(n_bins)])

        # One batched emulator call per spectrum, one row per bin.
        batch = {
            "omega_b": np.full(n_bins, omega_b),
            CDM_PARAM_NAME: omega_m - omega_b,
            "h": np.full(n_bins, self._get("h")),
            "tau_reio": np.full(n_bins, self._get("tau_reio")),
            "n_s": np.full(n_bins, self._get("n_s")),
            "ln10^{10}A_s": np.full(n_bins, self._ln10As()),
        }
        for name, (lo, hi) in _TRAINING_RANGE.items():
            v = batch[name]
            if np.any(v < lo) or np.any(v > hi):
                raise CosmoComputationError(
                    "%s = %s outside CosmoPower training range [%g, %g]"
                    % (name, np.array2string(v, precision=4), lo, hi))

        cl = {}
        for spec, (emu, is_log) in _EMU.items():
            pred = emu.predictions_np(batch)          # shape (n_bins, n_ell)
            if is_log:
                pred = 10.0 ** pred
            patched = np.full(_ELL.shape, np.nan)
            for i, m in enumerate(masks):
                patched[m] = pred[i, m]
            if np.isnan(patched).any():  # every multipole must belong to exactly one bin
                raise CosmoSevereError("l_switch bins leave multipoles uncovered: %s" % (l_switch,))
            cl[spec] = patched

        self._cl = cl
        self._lmax = lmax
        self._computed_pars = dict(self._pars)

    # -- output -------------------------------------------------------------
    def lensed_cl(self, lmax=-1, nofail=False):
        """Lensed C_l, dimensionless, arrays indexed by ell from 0 (like classy)."""
        if self._cl is None:
            self.compute()
        if lmax == -1:
            lmax = self._lmax
        lmax = int(lmax)
        if lmax > self._lmax:
            raise CosmoSevereError(
                "Can only compute up to lmax=%d (CosmoPower emulators stop at %d)"
                % (self._lmax, L_MAX_EMU))
        out = {"ell": np.arange(lmax + 1)}
        n = lmax - L_MIN_EMU + 1  # number of emulator multipoles to copy
        for spec, arr in self._cl.items():
            full = np.zeros(lmax + 1)
            if n > 0:
                full[L_MIN_EMU:lmax + 1] = arr[:n]
            out[spec] = full
        out["tp"] = np.zeros(lmax + 1)  # see caveat 3: not emulated
        return out

    def raw_cl(self, lmax=-1, nofail=False):
        raise CosmoSevereError(
            "raw_cl() (unlensed spectra) is not available: the CosmoPower CP_paper "
            "emulators are trained on LENSED spectra. Use lensed_cl(), and set "
            "unlensed_clTTTEEE = False in mock-CMB likelihoods.")

    # -- derived quantities that are plain functions of the inputs ----------
    def T_cmb(self):
        return _T_CMB

    def h(self):
        return self._get("h")

    def H0(self):
        return 100.0 * self._get("h")

    def n_s(self):
        return self._get("n_s")

    def tau_reio(self):
        return self._get("tau_reio")

    def omega_b(self):
        return self._get("omega_b")

    def Omega_b(self):
        return self._get("omega_b") / self._get("h") ** 2

    def Omega_m(self):
        # PLACEHOLDER: bin 0 only -- there is no single Omega_m in this model
        return self._get("omega_m0") / self._get("h") ** 2

    def get_current_derived_parameters(self, names):
        if not isinstance(names, list):
            raise TypeError("Deprecated")
        h = self._get("h")
        simple = {
            "h": lambda: h,
            "H0": lambda: 100.0 * h,
            "n_s": lambda: self._get("n_s"),
            "tau_reio": lambda: self._get("tau_reio"),
            "ln10^{10}A_s": self._ln10As,
            "A_s": lambda: np.exp(self._ln10As()) / 1e10,
            "omega_m": lambda: self._get("omega_m0"),        # placeholder, bin 0
            "Omega_m": lambda: self._get("omega_m0") / h**2,  # placeholder, bin 0
        }
        out = {}
        for name in names:
            if name not in simple:
                raise CosmoSevereError(
                    "derived parameter '%s' is not available in the CosmoPower "
                    "classy stand-in (no background / thermodynamics)" % name)
            out[name] = simple[name]()
        return out

    # -- things that have no meaning here ------------------------------------
    def _no_background(self, *args, **kwargs):
        raise CosmoSevereError(
            "background quantities (H(z), distances, sigma8, rs_drag, theta_s, ...) "
            "are not defined for the binned-omega_m CosmoPower model")

    Hubble = angular_distance = luminosity_distance = _no_background
    sigma8 = S8 = rs_drag = theta_s_100 = theta_star_100 = age = _no_background
    Omega_Lambda = Omega0_m = z_of_r = pk = pk_lin = _no_background
