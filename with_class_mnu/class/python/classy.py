"""
classy.py -- binned-omega_m model computed with REAL CLASS
===========================================================
(with_class_mnu version: one massive neutrino 0.06 eV + HMcode, to match the
Bolliet et al. 2023 lcdm v1 emulators in with_cp_lcdm/. Copy of
with_class/class/python/classy.py; changes are marked "[mnu]".)

Same interface and same model as the CosmoPower stand-in
(with_cosmopower/class/python/classy.py), but the spectra in each multipole
bin come from the real, compiled CLASS python wrapper instead of the emulator.
It exists so that the two pipelines can be benchmarked against each other
with everything else (MontePython, likelihood, param files, binning) identical.

How it works
  MontePython does `from classy import Class`. With this folder first on
  PYTHONPATH, that import lands here. This module then imports the real
  compiled `classy` (from site-packages) under the hood, and runs one real
  CLASS computation per omega_m bin.

The model (identical to the CosmoPower stand-in)
  Params: omega_b, h, tau_reio, n_s, ln10^{10}A_s (or A_s),
          omega_m0..omega_m{N-1}, l_switch1..l_switch{N}.
  Bin 0 is [2, l_switch1], bin i is (l_switch{i}, l_switch{i+1}].
  In bin i every spectrum (TT, TE, EE, PP, TP) is taken from CLASS run at
  [mnu] omega_cdm = omega_m{i} - omega_b - omega_nu, omega_nu = 0.00064420
  (one 0.06 eV neutrino), so omega_m is the TOTAL matter density
  (= CLASS Omega0_m*h^2), the same convention as with_cp_lcdm's stand-in.
  One bin (l_switch1 = l_max) = plain LCDM.

CLASS settings
  [mnu] Neutrinos and lensing set to the lcdm v1 emulators' training setup:
  N_ur = 2.0328, N_ncdm = 1, m_ncdm = 0.06 eV, non_linear = hmcode
  (see _PHYSICS below). Everything else is the CLASS default: BBN helium,
  HyRec, tanh reionization, T_cmb = 2.7255 K, default precision. Any key
  MontePython passes (e.g. via data.cosmo_arguments) is forwarded to CLASS
  and overrides these.

Differences from the CosmoPower stand-in (deliberate, small)
  * no training-box check (CLASS can compute anywhere it converges);
    CLASS failures raise CLASS's own CosmoComputationError -> step rejected.
  * 'tp' is the real CLASS T-phi cross spectrum, not zeros. MontePython's
    mock-CMB likelihood ignores it when neglect_TD = True.
  * 'bb' is left out, like in the stand-in, so both pipelines see the same keys.
  * the last l_switch must be >= l_max_scalars (the stand-in needs exactly 2508).

Speed: one full CLASS run per bin per MCMC step (several seconds each).
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Import the REAL classy, although this file is itself called classy.
# A compiled extension must be imported under the name 'classy' (its init
# symbol is PyInit_classy), so: step out of sys.modules / sys.path, import it,
# then put this module back.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_this = sys.modules.get(__name__)
_saved_path = list(sys.path)
try:
    del sys.modules[__name__]
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) != _HERE
                   and not os.path.isfile(os.path.join(p or os.getcwd(), "classy.py"))]
    import classy as _real  # noqa: E402  (the compiled CLASS wrapper)
finally:
    sys.path[:] = _saved_path
    sys.modules[__name__] = _this

# Either layout is fine: a compiled extension module (CLASS built from source)
# or the pip package (classy/__init__.py importing classy._classy).
if not hasattr(_real, "Class") or hasattr(_real, "_EMU") or \
        os.path.dirname(os.path.abspath(getattr(_real, "__file__", ""))) == _HERE:
    raise ImportError("could not find the real CLASS python wrapper; found %s"
                      % getattr(_real, "__file__", "?"))

CosmoError = getattr(_real, "CosmoError", Exception)
CosmoSevereError = _real.CosmoSevereError
CosmoComputationError = _real.CosmoComputationError
REAL_CLASSY_FILE = _real.__file__

L_MIN = 2


def _new_real_class():
    """A real CLASS instance. Recent pip classy finds its data files (BBN table,
    HyRec, ...) through importlib.resources.files('classy') when the instance is
    created, i.e. through sys.modules['classy'] -- which is this wrapper. Point
    that name at the real package for the duration of the constructor."""
    me = sys.modules.get(__name__)
    sys.modules[__name__] = _real
    try:
        return _real.Class()
    finally:
        sys.modules[__name__] = me

_OUR_KEYS = {"omega_b", "h", "tau_reio", "n_s", "ln10^{10}A_s", "A_s"}

# [mnu] physics of the lcdm v1 emulators (overridable from data.cosmo_arguments)
_PHYSICS = {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "non_linear": "hmcode"}
# [mnu] CLASS v3.4.1 omega_ncdm for one 0.06 eV neutrino (same as with_cp_lcdm)
OMEGA_NU = 0.00064420


def _is_bin_key(k):
    return (k.startswith("omega_m") and k[7:].isdigit()) or \
           (k.startswith("l_switch") and k[8:].isdigit())


class Class(object):

    def __init__(self, default=False):
        self._pars = {}
        self._codes = []            # one real CLASS instance per bin
        self._cl = None             # dict of arrays indexed by ell from 0
        self._computed_pars = None
        self._lmax = None

    # -- state ------------------------------------------------------------
    @property
    def state(self):
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
        for c in self._codes:
            c.empty()

    def struct_cleanup(self):
        self._cl = None
        self._computed_pars = None
        for c in self._codes:
            c.struct_cleanup()

    # -- parameter parsing (same rules as the CosmoPower stand-in) ---------
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
            raise CosmoSevereError("no l_switch1, l_switch2, ... parameters given")
        if any(b <= a for a, b in zip(vals[:-1], vals[1:])):
            raise CosmoSevereError("l_switch values must be strictly increasing: %s" % vals)
        return np.array(vals)

    def _ln10As(self):
        if "ln10^{10}A_s" in self._pars:
            return float(self._pars["ln10^{10}A_s"])
        if "A_s" in self._pars:
            return float(np.log(1e10 * float(self._pars["A_s"])))
        raise CosmoSevereError("missing scalar amplitude: give ln10^{10}A_s or A_s")

    # -- the computation ----------------------------------------------------
    def compute(self, level=None):
        if self._cl is not None and self._computed_pars == self._pars:
            return

        l_switch = self._l_switch()
        n_bins = len(l_switch)
        lmax = int(float(self._pars.get("l_max_scalars", l_switch[-1])))
        if l_switch[-1] < lmax:
            raise CosmoSevereError(
                "last l_switch (%g) is below l_max_scalars (%d): multipoles uncovered"
                % (l_switch[-1], lmax))

        omega_b = self._get("omega_b")
        omega_m = [self._get("omega_m%d" % i) for i in range(n_bins)]

        # what goes to CLASS: our cosmology + everything else MontePython passed
        base = dict(_PHYSICS)  # [mnu]
        base.update({k: v for k, v in self._pars.items()
                     if k not in _OUR_KEYS and not _is_bin_key(k)})
        base.update({
            "omega_b": omega_b,
            "h": self._get("h"),
            "tau_reio": self._get("tau_reio"),
            "n_s": self._get("n_s"),
            "ln10^{10}A_s": self._ln10As(),
            "l_max_scalars": lmax,
        })
        out = str(base.get("output", ""))
        for o in ("tCl", "pCl", "lCl"):
            if o not in out:
                out = (out + " " + o).strip()
        base["output"] = out
        base["lensing"] = "yes"

        while len(self._codes) < n_bins:
            self._codes.append(_new_real_class())

        ell = np.arange(lmax + 1)
        edges = [L_MIN - 1] + list(l_switch)
        cl = {k: np.zeros(lmax + 1) for k in ("tt", "ee", "te", "pp", "tp")}
        done = {}  # bins with the same omega_m share one CLASS run
        for i in range(n_bins):
            mask = (ell > edges[i]) & (ell <= edges[i + 1])
            if not mask.any():
                continue
            key = omega_m[i]
            if key not in done:
                c = self._codes[i]
                c.struct_cleanup()
                c.empty()
                c.set(dict(base, omega_cdm=omega_m[i] - omega_b - OMEGA_NU))  # [mnu]
                c.compute(level) if level is not None else c.compute()
                done[key] = c.lensed_cl(lmax)
            res = done[key]
            for k in cl:
                cl[k][mask] = res[k][mask]

        covered = np.zeros(lmax + 1, bool)
        for i in range(n_bins):
            covered |= (ell > edges[i]) & (ell <= edges[i + 1])
        if not covered[L_MIN:].all():
            raise CosmoSevereError("l_switch bins leave multipoles uncovered: %s" % l_switch)

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
            raise CosmoSevereError("Can only compute up to lmax=%d" % self._lmax)
        out = {"ell": np.arange(lmax + 1)}
        for k, arr in self._cl.items():
            out[k] = arr[:lmax + 1].copy()
        return out

    def raw_cl(self, lmax=-1, nofail=False):
        raise CosmoSevereError(
            "raw_cl() is not provided by the binned wrapper (the CosmoPower "
            "stand-in can't provide it either). Use lensed_cl().")

    # -- derived quantities (same set as the CosmoPower stand-in) -----------
    def T_cmb(self):
        if self._codes and self._cl is not None:
            return self._codes[0].T_cmb()
        return 2.7255

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
        # PLACEHOLDER: bin 0 only, same convention as the stand-in
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
            "omega_m": lambda: self._get("omega_m0"),
            "Omega_m": lambda: self._get("omega_m0") / h**2,
        }
        out = {}
        for name in names:
            if name not in simple:
                raise CosmoSevereError(
                    "derived parameter '%s' is not available in the binned model" % name)
            out[name] = simple[name]()
        return out

    def _no_background(self, *args, **kwargs):
        raise CosmoSevereError(
            "background quantities are not defined for the binned-omega_m model")

    Hubble = angular_distance = luminosity_distance = _no_background
    sigma8 = S8 = rs_drag = theta_s_100 = theta_star_100 = age = _no_background
    Omega_Lambda = Omega0_m = z_of_r = pk = pk_lin = _no_background
