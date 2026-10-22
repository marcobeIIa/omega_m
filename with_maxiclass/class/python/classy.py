"""
classy.py -- binned-omega_m model on top of mAxiCLASS (multi-field EDE)
=======================================================================
(with_maxiclass version. Copy of with_class_mnu/class/python/classy.py with
the changes marked "[maxi]". MOCK MAKING ONLY: this pipeline is never sampled
with MontePython; its mocks are fitted with the other pipelines through
MODEL_FROM=with_maxiclass in slurm/run_chains.sbatch.)

Layout
  with_maxiclass/maxiclass/            git clone of mAxiCLASS (branch mAxiCLASShoot),
                                       compiled in place (see with_maxiclass/README.md)
  with_maxiclass/class/python/classy.py  this file
  with_maxiclass/model/                mocks (make them with models/make_fiducial.py
                                       --pipeline with_maxiclass)

The model (same as with_class_mnu)
  Params: omega_b, h, tau_reio, n_s, ln10^{10}A_s (or A_s),
          omega_m0..omega_m{N-1}, l_switch1..l_switch{N}.
  Bin 0 is [2, l_switch1], bin i is (l_switch{i}, l_switch{i+1}].
  omega_cdm = omega_m{i} - omega_b - OMEGA_NU (one 0.06 eV neutrino), so
  omega_m is the TOTAL matter density b + cdm + nu, as in with_class_mnu and
  with_cp_lcdm. The EDE fields are NOT part of omega_m (CLASS's Omega0_m
  counts their matter-like part rho-3p, which is ~1e-10 today).
  One bin (l_switch1 = l_max) = plain LCDM, or plain EDE if EDE keys are set.

[maxi] Beyond-LCDM physics
  Every key that is not one of ours is forwarded to mAxiCLASS, e.g. for the
  2-axion EDE of Bella et al. 2026 (arXiv:2604.13535):
      N_mscf = 2, n_axion_mscf = '3,3', theta_ini_mscf = '2.8,2.8',
      fraction_maxion_ac = '0.085,0.073', log10_maxion_ac = '-3.440,-4.000'
  When N_mscf > 0, _EDE_DEFAULTS below are added unless given explicitly:
  shooting on (f_EDE and a_c are targets), theta' = 0, no attractor IC, and
  shooting tolerances 1e-3. The repo's example .ini files use tolerance 1,
  which stops the shooting after one step (f_EDE 0.138 for a 0.10 target in
  a test); 1e-2 already hits the target. After compute() the achieved peak
  f_EDE and log10 a_c of each field are read off the background and compared
  with the targets (ede_summary()); a miss raises CosmoComputationError.

CLASS settings
  N_ur = 2.0328, N_ncdm = 1, m_ncdm = 0.06 eV, non_linear = hmcode
  (_PHYSICS, same as with_class_mnu); everything else is the mAxiCLASS
  default. Any key passed overrides these.

Finding mAxiCLASS
  The compiled extension is loaded BY PATH from
  $MAXICLASS_DIR (default: with_maxiclass/maxiclass)/python/build/lib*/classy*.so
  (or python/classy*.so after build_ext --inplace), so a pip classy in the
  same environment (Hive's 'class' env has classy 3.4.1) is never picked up.
  mAxiCLASS finds its data files (BBN, HyRec) through the path compiled into
  it, so compile it on the machine that runs it.

  mAxiCLASS prints debug lines from C; they are hidden during compute()
  unless MAXICLASS_VERBOSE=1.
"""

import glob
import importlib.machinery
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# [maxi] Load the compiled mAxiCLASS extension by path, under the name
# 'classy' (its init symbol is PyInit_classy), then put this module back.
# ---------------------------------------------------------------------------
MAXICLASS_DIR = os.path.abspath(os.environ.get(
    "MAXICLASS_DIR", os.path.join(_HERE, "..", "..", "maxiclass")))


def _find_extension():
    pats = []
    for suf in importlib.machinery.EXTENSION_SUFFIXES:
        pats += [os.path.join(MAXICLASS_DIR, "python", "build", "lib*", "classy" + suf),
                 os.path.join(MAXICLASS_DIR, "python", "classy" + suf)]
    found = sorted({f for p in pats for f in glob.glob(p)}, key=os.path.getmtime)
    if not found:
        raise ImportError(
            "no compiled mAxiCLASS python extension for this python (%s) under %s/python. "
            "Build it: cd %s && make -j libclass.a && cd python && python setup.py build"
            % (sys.version.split()[0], MAXICLASS_DIR, MAXICLASS_DIR))
    return found[-1]   # newest build


REAL_CLASSY_FILE = _find_extension()
_this = sys.modules.get(__name__)
try:
    sys.modules.pop(__name__, None)
    _spec = importlib.util.spec_from_file_location("classy", REAL_CLASSY_FILE)
    _real = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_real)
finally:
    sys.modules[__name__] = _this

CosmoError = getattr(_real, "CosmoError", Exception)
CosmoSevereError = _real.CosmoSevereError
CosmoComputationError = _real.CosmoComputationError

L_MIN = 2

_OUR_KEYS = {"omega_b", "h", "tau_reio", "n_s", "ln10^{10}A_s", "A_s"}

# physics of the lcdm v1 emulators / with_class_mnu (overridable)
_PHYSICS = {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "non_linear": "hmcode"}
# CLASS v3.4.1 omega_ncdm for one 0.06 eV neutrino (same as with_class_mnu)
OMEGA_NU = 0.00064420

# [maxi] added when N_mscf > 0 (overridable)
_EDE_DEFAULTS = {"do_shooting": "yes", "do_shooting_mscf": "yes",
                 "attractor_ic_scf": "no",
                 "tol_shooting_deltaF": 1e-3, "tol_shooting_deltax": 1e-3,
                 "background_Nloga": 100000}
# Tests (2-axion PDH0 mean): tolerance 1e-4 makes the Newton shooting fail;
# background_Nloga default -> 1e5 changes the mock by sqrt(chi2) 0.13,
# 1e5 -> 3e5 by 0.015 (Planck-like noise, l >= 30).
# [maxi] shooting check: allowed misses of the achieved peak vs the targets
F_EDE_RTOL = 0.01
LOG10_AC_ATOL = 0.01

_QUIET = {"input_verbose": 0, "background_verbose": 0, "thermodynamics_verbose": 0,
          "perturbations_verbose": 0, "transfer_verbose": 0, "primordial_verbose": 0,
          "harmonic_verbose": 0, "fourier_verbose": 0, "lensing_verbose": 0,
          "output_verbose": 0}


def _is_bin_key(k):
    return (k.startswith("omega_m") and k[7:].isdigit()) or \
           (k.startswith("l_switch") and k[8:].isdigit())


def _floats(v):
    """'0.085,0.073' / 0.085 / [0.085, 0.073] -> [0.085, 0.073]"""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    return [float(x) for x in str(v).split(",") if x.strip()]


class _CStdoutSilenced(object):
    """Hide what the C code prints to stdout (fd 1)."""

    def __enter__(self):
        self.on = os.environ.get("MAXICLASS_VERBOSE", "0") in ("", "0")
        if self.on:
            sys.stdout.flush()
            self.saved = os.dup(1)
            self.null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.null, 1)
        return self

    def __exit__(self, *exc):
        if self.on:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            os.dup2(self.saved, 1)
            os.close(self.saved)
            os.close(self.null)
        return False


class Class(object):

    def __init__(self, default=False):
        self._pars = {}
        self._codes = []            # one mAxiCLASS instance per bin
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

    # -- parameter parsing -------------------------------------------------
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

    def _n_mscf(self):
        return int(float(self._pars.get("N_mscf", 0)))

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

        # what goes to CLASS: our cosmology + everything else that was passed
        base = dict(_PHYSICS)
        if self._n_mscf() > 0:                                        # [maxi]
            base.update(_EDE_DEFAULTS)
        if os.environ.get("MAXICLASS_VERBOSE", "0") in ("", "0"):     # [maxi]
            base.update(_QUIET)
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
            self._codes.append(_real.Class())

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
                c.set(dict(base, omega_cdm=omega_m[i] - omega_b - OMEGA_NU))
                with _CStdoutSilenced():                              # [maxi]
                    c.compute(level) if level is not None else c.compute()
                # [maxi] MP-branch of mAxiCLASS never reads 'h' (only H0) and silently
                # falls back to h = 0.6781: refuse such a build instead of using it.
                if abs(c.h() / base["h"] - 1) > 1e-10:
                    raise CosmoSevereError(
                        "mAxiCLASS ignored h = %.6g (computed with h = %.6g): wrong branch? "
                        "use mAxiCLASShoot (see with_maxiclass/check_install.py)"
                        % (base["h"], c.h()))
                self._check_shooting(c)                               # [maxi]
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

    # -- [maxi] EDE diagnostics ----------------------------------------------
    @staticmethod
    def _summary_of(c, n):
        bg = c.get_background()
        loga = np.log10(1.0 / (1.0 + bg["z"]))
        rows = []
        for k in range(n):
            f = bg["(.)rho_mscf[%d]" % k] / bg["(.)rho_crit"]
            j = int(np.argmax(f))
            rows.append({"field": k, "f_peak": float(f[j]), "log10_a_peak": float(loga[j])})
        return rows

    def ede_summary(self, bin_index=0):
        """Per field: peak f_EDE = rho_i/rho_tot and log10 a at the peak (bin 0's run)."""
        if self._cl is None:
            self.compute()
        return self._summary_of(self._codes[bin_index], self._n_mscf())

    def _check_shooting(self, c):
        n = self._n_mscf()
        if n == 0:
            return
        rows = self._summary_of(c, n)
        f_t = _floats(self._pars.get("fraction_maxion_ac", []))
        a_t = _floats(self._pars.get("log10_maxion_ac", []))
        for r in rows:
            k = r["field"]
            if k < len(f_t) and abs(r["f_peak"] / f_t[k] - 1) > F_EDE_RTOL:
                raise CosmoComputationError(
                    "EDE shooting missed f_EDE of field %d: %.4g vs target %.4g"
                    % (k, r["f_peak"], f_t[k]))
            if k < len(a_t) and abs(r["log10_a_peak"] - a_t[k]) > LOG10_AC_ATOL:
                raise CosmoComputationError(
                    "EDE shooting missed log10 a_c of field %d: %.4f vs target %.4f"
                    % (k, r["log10_a_peak"], a_t[k]))

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
        raise CosmoSevereError("raw_cl() is not provided by the binned wrapper. Use lensed_cl().")

    # -- derived quantities ---------------------------------------------------
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
        # PLACEHOLDER: bin 0 only, same convention as the other pipelines
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

    # [maxi] background quantities of the bin-0 mAxiCLASS run (useful for mocks:
    # theta_s, sigma8, ... of the EDE cosmology). Not defined for >1 bin.
    def bin0(self):
        """The mAxiCLASS instance of bin 0 (after compute())."""
        if self._cl is None:
            self.compute()
        return self._codes[0]
