"""
classy.py -- CosmoPower-backed stand-in for CLASS's Python wrapper
==================================================================
(with_cp_lcdm version: Bolliet et al. 2023 LCDM emulators, CLASS-trained)
[poly] + optional smooth delta omega_m(l) (linear response), see below.

MontePython does `from classy import Class` and then drives the object with
set() / compute() / lensed_cl() / struct_cleanup() / empty().  This module
provides a `Class` with that same interface, but the spectra come from the
pretrained CosmoPower LCDM emulators of Bolliet et al. 2023
(arXiv:2303.01591, github.com/cosmopower-organization/lcdm, version v1),
evaluated piecewise in ell so that omega_m can take a different value in
each multipole bin.

This file is the with_cosmopower/class/python/classy.py stand-in with only
the emulator swapped. Differences from that file are marked "[lcdm v1]".

What the emulators are (checked against CLASS v3.4.1, Planck-2018 LCDM,
1 massive nu 0.06 eV + N_ur 2.0328, non_linear hmcode, 2026-10-16):
    TTTEEE/TT_v1   cosmopower_NN        -> log10 l(l+1)C_l^TT/2pi  (lensed)
    TTTEEE/EE_v1   cosmopower_NN        -> log10 l(l+1)C_l^EE/2pi  (lensed)
    TTTEEE/TE_v1   cosmopower_PCAplusNN ->       l(l+1)C_l^TE/2pi  (lensed;
                                                 not logged, TE changes sign)
    PP/PP_v1       cosmopower_NN        -> log10 [l(l+1)]^2 C_l^phiphi
    all dimensionless, every multipole 2 <= l <= 11000,
    inputs: omega_b, omega_cdm, H0, tau_reio, n_s, ln10^{10}A_s (PP: no tau).
    The stand-in divides these factors out, so lensed_cl() returns plain C_l.
Agreement with that CLASS (l 30-2508): TT 0.1%, EE 0.1%; PP 0.2% for
L < 1000, rising to ~1-2% at L 1000-2508 (non-linear lensing version).
The emulators include ONE MASSIVE NEUTRINO (0.06 eV) and HMcode lensing:
a CLASS mock to be fitted with this stand-in must be made with
N_ur 2.0328, N_ncdm 1, m_ncdm 0.06, non_linear hmcode.

Faithful to real classy (so MontePython's Planck / mock-CMB likelihoods work):
  * lensed_cl(lmax=-1) returns a dict of arrays indexed by ell from 0
    (entries at l=0,1 are zero), keys 'ell','tt','ee','te','pp','tp',
    dimensionless -- MontePython multiplies by (T_cmb*1e6)^2 itself.
  * lmax=-1 means "up to l_max_scalars" (or the last l_switch if not set);
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
     [lcdm v1] l_switch{N} must be >= l_max_scalars (and <= 11000); the
     emulators go to l = 11000, so l_switch{N} = 2508 with l_max 2508 works
     exactly as before.  In bin i every spectrum (TT, TE, EE and PP) is taken
     from the emulator evaluated at
         [lcdm v1] omega_cdm = omega_m{i} - omega_b - omega_nu,
     omega_nu = 0.00064420 (CLASS value for one 0.06 eV neutrino), so that
     omega_m is the TOTAL matter density, the same as CLASS's Omega0_m*h^2.
     H0 = 100*h.  Applying the same ell-bins to PP is a modelling choice: the
     lensing potential's L is not the CMB's l.  For a single bin (N=1,
     l_switch1=2508) the model is exactly LCDM.
  2. raw_cl() (UNlensed spectra) raises: no unlensed emulator is loaded.
     MontePython's Planck and mock-CMB likelihoods use lensed_cl().
  3. The emulators provide neither lensed BB nor the T-phi cross.
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
  5. [lcdm v1] Points outside the emulator training box raise
     CosmoComputationError (the step is rejected), per bin:
        omega_b [0.0193,0.0253], omega_cdm [0.0799,0.2], H0 [40.05,100],
        tau_reio [0.02,0.12], n_s [0.881,1.049], ln10^{10}A_s [2.5,3.5].
     (Box inferred from the stored parameter mean/std, assuming uniform
     sampling.)  With omega_b ~ 0.0224 this means omega_m bins in about
     [0.103, 0.223]; keep the .param priors inside.

[poly] Smooth delta omega_m(l), linear response (optional, 2026-10-16)
  If any of dwm_a, dwm_b, dwm_c, dwm_d, dwm_e is passed, every spectrum
  (TT, TE, EE, PP) becomes
      C_l = C_l(base) + dC_l/domega_m|_theta_s * delta_omega_m(l),
      delta_omega_m(l) = dwm_a + dwm_b x + dwm_c x^2 (+ dwm_d x^3 + dwm_e x^4),
      x = (l - l_c) / l_norm.
  Extra params: l_c (required), l_norm (default 1000), dlnh_dlnomega_m
  (optional constant; default: CLASS fit, see below). Missing coefficients
  are 0.
  * C_l(base) is the usual model above (1 bin = plain LCDM at omega_m0).
  * The derivative is taken at the CURRENT point (in each bin, at that bin's
    omega_m), at fixed 100*theta_s, omega_b, tau_reio, n_s, A_s, by the chain
        dC/domega_m|_theta = dC/domega_cdm|_H0 + dC/dH0 * dH0/domega_m|_theta,
        dH0/domega_m|_theta = 100 h/omega_m * dlnh_dlnomega_m,
    as ONE exact forward-mode derivative (Jacobian-vector product) through the
    networks along that direction -- analytic, no finite-difference step.
    dlnh_dlnomega_m = dln h/dln omega_m at fixed theta_s and omega_b (the
    networks have no theta_s): a fit to CLASS 3.4.1 (1 nu 0.06 eV) in
    (omega_b, omega_m, h), good to 0.3% over the prior box (-0.748 at
    omega_b 0.02237, omega_m 0.143, h 0.674; -0.738 at the omega_m 0.15 mock).
  * dwm_a is exactly degenerate (to first order) with omega_m0 plus the
    matching h shift: keep dwm_a fixed to 0 when omega_m0 is free. Then
    omega_m(l_c) = omega_m0 and dwm_b, dwm_c describe the l-dependence.
  * Linear in delta_omega_m. Against the exact model (emulator evaluated at
    omega_m(l), h on the CLASS fixed-theta_s curve), sqrt(chi2) of the error
    with the fake_planck_realistic noise: constant 1/3/5 sigma(omega_m) ->
    0.01/0.11/0.29. BUT a free quadratic is weakly constrained at high l
    (Fisher sigma of omega_m(l): 0.005 at l 1600, 0.018 at 2500), and at
    (dwm_b, dwm_c) on the 2-sigma Fisher ellipse the error is ~3 (as large as
    the signal): far from b = c = 0 this is a phenomenological model, not
    "omega_m at that l". The perturbation itself is not box-checked.
  * Without dwm_* keys the code path is the old one, bit for bit.

Configuration (environment variables, optional):
    COSMOPOWER_CMB_DIR   [lcdm v1] directory holding TTTEEE/ and PP/ of the
                         cosmopower-organization/lcdm repo (default:
                         ../../emulators relative to this file, i.e.
                         with_cp_lcdm/emulators)
    COSMOPOWER_CDM_PARAM_NAME   emulator's CDM input name (default omega_cdm)
"""

import os
import numpy as np
from cosmopower import cosmopower_NN, cosmopower_PCAplusNN

# ---------------------------------------------------------------------------
# Emulators (loaded once, at import)
# ---------------------------------------------------------------------------
# [lcdm v1] Default: the lcdm repo cloned next to this class/ folder
# (with_cp_lcdm/class/python/classy.py -> with_cp_lcdm/emulators/...).
CP_CMB_DIR = os.environ.get(
    "COSMOPOWER_CMB_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "emulators")),
)
CDM_PARAM_NAME = os.environ.get("COSMOPOWER_CDM_PARAM_NAME", "omega_cdm")

# [lcdm v1] name -> (network class, file stem, output is log10?, output form)
#   form "dl": emulator gives l(l+1)C_l/2pi ; form "l4": gives [l(l+1)]^2 C_l
_NETWORKS = {
    "tt": (cosmopower_NN, os.path.join("TTTEEE", "TT_v1"), True, "dl"),
    "ee": (cosmopower_NN, os.path.join("TTTEEE", "EE_v1"), True, "dl"),
    "te": (cosmopower_PCAplusNN, os.path.join("TTTEEE", "TE_v1"), False, "dl"),
    "pp": (cosmopower_NN, os.path.join("PP", "PP_v1"), True, "l4"),
}
_EMU = {
    k: (cls(restore=True, restore_filename=os.path.join(CP_CMB_DIR, stem)), is_log, form)
    for k, (cls, stem, is_log, form) in _NETWORKS.items()
}

_ELL = np.asarray(_EMU["tt"][0].modes, dtype=int)  # [lcdm v1] 2..11000
for _k, (_e, _, _) in _EMU.items():
    if not np.array_equal(np.asarray(_e.modes, dtype=int), _ELL):
        raise ImportError("CosmoPower network '%s' has a different ell grid than TT" % _k)
L_MIN_EMU, L_MAX_EMU = int(_ELL[0]), int(_ELL[-1])
# lensed_cl() maps emulator array index i -> multipole L_MIN_EMU + i. That is only
# valid if the emulator grid is every integer multipole with no gaps; refuse to
# load otherwise (this is exactly the kind of silent ell shift we must never have).
if not np.array_equal(_ELL, np.arange(L_MIN_EMU, L_MAX_EMU + 1)):
    raise ImportError("CosmoPower ell grid is not contiguous: %s ... %s" % (_ELL[:5], _ELL[-5:]))

# [lcdm v1] factor that turns the emulator output into plain C_l, per form
_LL1 = _ELL * (_ELL + 1.0)
_TO_CL = {"dl": 2.0 * np.pi / _LL1, "l4": 1.0 / _LL1 ** 2}

# [lcdm v1] training box of the v1 networks (TT/TE/EE; PP has no tau_reio)
_TRAINING_RANGE = {
    "omega_b": (0.0193, 0.0253),
    CDM_PARAM_NAME: (0.0799, 0.2),
    "H0": (40.05, 100.0),
    "tau_reio": (0.02, 0.12),
    "n_s": (0.881, 1.049),
    "ln10^{10}A_s": (2.5, 3.5),
}

# [lcdm v1] one massive neutrino, 0.06 eV: CLASS v3.4.1 omega_ncdm
# (Omega0_m*h^2 - omega_b - omega_cdm at the Planck point).
OMEGA_NU = 0.00064420

_T_CMB = 2.7255  # K; CLASS default

# [poly] polynomial coefficients of delta_omega_m(l), in order of power of x
_POLY_KEYS = ("dwm_a", "dwm_b", "dwm_c", "dwm_d", "dwm_e")
_L_NORM_DEFAULT = 1000.0


def _dlnh_dlnomega_m(omega_b, omega_m, h):
    """[poly] dln h/dln omega_m at fixed 100*theta_s and omega_b.

    Fit (linear in logs) to real CLASS 3.4.1 (N_ur 2.0328 + 1 nu 0.06 eV,
    theta_s_100 by central differences) on a grid over omega_b [0.0193,
    0.0253], omega_m [0.11, 0.21], h [0.60, 0.75]; checked at 25 random
    points in that box: max error 0.26%, rms 0.13%. (A constant -0.748 would
    be off by up to 19% at the box edges; ~1.4% at omega_m 0.15.)
    """
    return (-0.74846 - 0.05312 * np.log(omega_b / 0.02237)
            + 0.21286 * np.log(omega_m / 0.143) + 0.50894 * np.log(h / 0.674))


def _forward_and_jvp(emu, x, v):
    """[poly] Network output y(x) and its directional derivative dy/dx . v.

    Same forward pass as cosmopower's forward_pass_np (written with the same
    floating-point expressions, so y is bit-identical), carrying the tangent
    along: exact (analytic) derivative of the network, no step size.
    x, v: (n_rows, n_params) in the network's own parameter order.
    """
    layer = (x - emu.parameters_mean_) / emu.parameters_std_
    dlayer = v / emu.parameters_std_
    for i in range(emu.n_layers - 1):
        act = np.dot(layer, emu.W_[i]) + emu.b_[i]
        dact = np.dot(dlayer, emu.W_[i])
        alpha, beta = emu.alphas_[i], emu.betas_[i]
        e = np.exp(-alpha * act)
        layer = (beta + (1. - beta) * 1. / (1. + e)) * act
        sig = 1. / (1. + e)
        # d/da [(beta + (1-beta) sig(alpha a)) a]
        dlayer = (beta + (1. - beta) * sig * (1. + alpha * act * (1. - sig))) * dact
    out = np.dot(layer, emu.W_[-1]) + emu.b_[-1]
    dout = np.dot(dlayer, emu.W_[-1])
    if hasattr(emu, "pca_transform_matrix_"):  # cosmopower_PCAplusNN
        y = np.dot(out * emu.pca_std_ + emu.pca_mean_,
                   emu.pca_transform_matrix_) * emu.features_std_ + emu.features_mean_
        dy = np.dot(dout * emu.pca_std_, emu.pca_transform_matrix_) * emu.features_std_
    else:                                      # cosmopower_NN
        y = out * emu.features_std_ + emu.features_mean_
        dy = dout * emu.features_std_
    return y, dy


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
    # [lcdm v1] the emulators reach l = 11000: the last l_switch only has to fit
    # inside that (compute() checks it also reaches l_max_scalars).
    if key[-1] > L_MAX_EMU:
        raise CosmoSevereError(
            "the last l_switch must be <= the emulator's l_max = %d (got %g)"
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
        self._dcl = None            # [poly] dC_l/domega_m|theta_s on _ELL
        self._dwm = None            # [poly] delta_omega_m(l) on _ELL

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
        self._dcl = None
        self._dwm = None

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
                "l_switch1 = l_max for plain LCDM)")
        return np.array(vals)

    def _ln10As(self):
        if "ln10^{10}A_s" in self._pars:
            return float(self._pars["ln10^{10}A_s"])
        if "A_s" in self._pars:
            return float(np.log(1e10 * float(self._pars["A_s"])))
        raise CosmoSevereError("missing scalar amplitude: give ln10^{10}A_s or A_s")

    def _requested_lmax(self, l_switch):
        # [lcdm v1] default l_max = last l_switch (was: the emulator's l_max)
        lmax = int(float(self._pars.get("l_max_scalars", l_switch[-1])))
        if lmax > L_MAX_EMU:
            raise CosmoSevereError(
                "l_max_scalars = %d requested, but the CosmoPower emulators stop at "
                "l = %d. Lower l_max in the likelihood's .data file." % (lmax, L_MAX_EMU))
        if lmax > l_switch[-1]:
            raise CosmoSevereError(
                "l_max_scalars = %d is above the last l_switch = %g: multipoles "
                "%g < l <= %d would belong to no bin" % (lmax, l_switch[-1], l_switch[-1], lmax))
        return lmax

    def _poly(self):
        """[poly] (coefs, l_c, l_norm, dlnh_dlnomega_m), or None if off."""
        if not any(k in self._pars for k in _POLY_KEYS):
            return None
        coefs = [float(self._pars.get(k, 0.0)) for k in _POLY_KEYS]
        while len(coefs) > 1 and coefs[-1] == 0.0:
            coefs.pop()
        if "l_c" not in self._pars:
            raise CosmoSevereError(
                "dwm_* coefficients given but no pivot l_c: delta_omega_m(l) = "
                "dwm_a + dwm_b x + dwm_c x^2 with x = (l - l_c)/l_norm")
        l_norm = float(self._pars.get("l_norm", _L_NORM_DEFAULT))
        if l_norm <= 0:
            raise CosmoSevereError("l_norm must be > 0 (got %g)" % l_norm)
        # None -> fitted CLASS slope at the current point (per bin)
        slope = self._pars.get("dlnh_dlnomega_m", None)
        slope = None if slope is None else float(slope)
        return coefs, float(self._pars["l_c"]), l_norm, slope

    # -- the computation ----------------------------------------------------
    def compute(self, level=None):
        # Nothing changed since the last compute -> keep the spectra.
        if self._cl is not None and self._computed_pars == self._pars:
            return

        l_switch = self._l_switch()
        n_bins = len(l_switch)
        masks = _bin_masks(l_switch)
        lmax = self._requested_lmax(l_switch)

        omega_b = self._get("omega_b")
        omega_m = np.array([self._get("omega_m%d" % i) for i in range(n_bins)])

        # One batched emulator call per spectrum, one row per bin.
        batch = {
            "omega_b": np.full(n_bins, omega_b),
            CDM_PARAM_NAME: omega_m - omega_b - OMEGA_NU,          # [lcdm v1]
            "H0": np.full(n_bins, 100.0 * self._get("h")),         # [lcdm v1]
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

        # [poly] smooth delta omega_m(l) on, if any coefficient is given
        poly = self._poly()
        if poly is not None:
            coefs, l_c, l_norm, slope = poly
            x = (_ELL - l_c) / l_norm
            dwm = np.zeros(_ELL.shape)
            for k, c in enumerate(coefs):
                dwm += c * x ** k
            # tangent direction in emulator inputs: d omega_m = 1 at fixed
            # theta_s -> d omega_cdm = 1, dH0 = 100 h/omega_m * dlnh/dlnomega_m
            tangent = {name: np.zeros(n_bins) for name in batch}
            tangent[CDM_PARAM_NAME] = np.ones(n_bins)
            if slope is None:
                slope = _dlnh_dlnomega_m(omega_b, omega_m, batch["H0"] / 100.0)
            tangent["H0"] = batch["H0"] / omega_m * slope

        # [lcdm v1] only multipoles up to lmax are used / checked
        n_use = lmax - L_MIN_EMU + 1
        cl, dcl = {}, {}
        for spec, (emu, is_log, form) in _EMU.items():
            if poly is None:
                pred = emu.predictions_np(batch)      # shape (n_bins, n_ell)
                dpred = None
            else:                                     # [poly] output + derivative
                pred, dpred = _forward_and_jvp(emu, emu.dict_to_ordered_arr_np(batch),
                                               emu.dict_to_ordered_arr_np(tangent))
            if is_log:
                pred = 10.0 ** pred
                if dpred is not None:
                    dpred = pred * np.log(10.0) * dpred   # [poly] d(10^y) = 10^y ln10 dy
            pred = pred * _TO_CL[form]                # [lcdm v1] -> plain C_l
            patched = np.full(_ELL.shape, np.nan)
            for i, m in enumerate(masks):
                patched[m] = pred[i, m]
            if np.isnan(patched[:n_use]).any():  # every multipole must belong to exactly one bin
                raise CosmoSevereError("l_switch bins leave multipoles uncovered: %s" % (l_switch,))
            if dpred is not None:                     # [poly] add the linear response
                dpred = dpred * _TO_CL[form]
                dpatched = np.full(_ELL.shape, np.nan)
                for i, m in enumerate(masks):
                    dpatched[m] = dpred[i, m]
                dcl[spec] = dpatched[:n_use]
                patched = patched + dpatched * dwm
            cl[spec] = patched[:n_use]

        self._cl = cl
        self._dcl = dcl if poly is not None else None
        self._dwm = dwm[:n_use] if poly is not None else None
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

    def dcl_domega_m(self, lmax=-1):
        """[poly] (not in real classy) dC_l/domega_m at fixed theta_s used by
        the polynomial model, same layout as lensed_cl (no 'tp'), plus
        'delta_omega_m' = delta_omega_m(l). Needs a dwm_* key to be set."""
        if self._cl is None:
            self.compute()
        if self._dcl is None:
            raise CosmoSevereError("dcl_domega_m(): no dwm_* coefficient set")
        lmax = self._lmax if lmax == -1 else int(lmax)
        if lmax > self._lmax:
            raise CosmoSevereError("Can only compute up to lmax=%d" % self._lmax)
        out = {"ell": np.arange(lmax + 1)}
        n = lmax - L_MIN_EMU + 1
        for spec, arr in list(self._dcl.items()) + [("delta_omega_m", self._dwm)]:
            full = np.zeros(lmax + 1)
            if n > 0:
                full[L_MIN_EMU:lmax + 1] = arr[:n]
            out[spec] = full
        return out

    def raw_cl(self, lmax=-1, nofail=False):
        raise CosmoSevereError(
            "raw_cl() (unlensed spectra) is not available: the CosmoPower lcdm v1 "
            "emulators loaded here are LENSED spectra. Use lensed_cl(), and set "
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
