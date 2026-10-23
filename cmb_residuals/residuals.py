"""
CMB residuals of fitted models with respect to a mock.

Used by cmb_residuals.ipynb (same folder). Everything is resolved relative to
the omega_m/ root (the parent of this folder), so it works on the Mac and on
Hive whatever the notebook's working directory.

  Run(folder)        one MontePython output folder (log.param + chains)
    .bestfit()       classy inputs at the minimum -lnL sample
    .draws(n)        classy inputs of n posterior samples (weighted, after burn-in)
  spectra(run, p)    lensed TT, EE, TE [muK^2] and DD = l(l+1) C_l^pp, l = l_min..l_max,
                     computed with the pipeline the run was sampled with
                     (classy.py under the log.param's data.path['cosmo'])
  Mock(path, run)    mock .dat (MontePython mock-CMB format) + the run's noise
  residual(...)      per-l or binned residual of a spectrum w.r.t. the mock
"""

import ast
import glob
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # omega_m/
SPECTRA = ("tt", "te", "ee", "dd")


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def to_local(path):
    """A path written on another machine (e.g. /home/mcbella/omega_m/with_cp_lcdm/class)
    -> the same file under this omega_m/ root. Relative paths are taken w.r.t. ROOT."""
    p = Path(path)
    if not p.is_absolute():
        return ROOT / p
    if p.exists():
        return p
    parts = p.parts
    if "omega_m" in parts:
        return ROOT.joinpath(*parts[parts.index("omega_m") + 1:])
    for i, part in enumerate(parts):                 # fallback: pipeline / MP folder name
        if part.startswith("with_") or part == "montepython_fake":
            return ROOT.joinpath(*parts[i:])
    raise FileNotFoundError(f"can't map {path} to the local omega_m root {ROOT}")


# ---------------------------------------------------------------------------
# log.param
# ---------------------------------------------------------------------------
def read_log_param(folder):
    """Parameters, cosmo arguments, cosmo path and likelihood settings of a run."""
    text = (Path(folder) / "log.param").read_text()
    pars = {n: ast.literal_eval(v) for n, v in re.findall(
        r"^data\.parameters\[['\"](.+?)['\"]\]\s*=\s*(\[.*?\])", text, re.M)}
    cosmo_args = {}
    for block in re.findall(r"^data\.cosmo_arguments\.update\((\{.*?\})\)", text, re.M):
        cosmo_args.update(ast.literal_eval(block))
    cosmo_path = re.findall(r"^data\.path\['cosmo'\]\s*=\s*'(.+?)'", text, re.M)[-1]
    exp = ast.literal_eval(re.search(r"^data\.experiments\s*=\s*(\[.*?\])", text, re.M).group(1))
    if len(exp) != 1:
        raise ValueError(f"{folder}: expected one likelihood, got {exp}")
    lik = {}
    for k, v in re.findall(rf"^{exp[0]}\.(\w+)\s*=\s*(.+?)\s*$", text, re.M):
        if k == "data_directory" and v.startswith("data.path"):
            continue                                 # the .data default, overridden below
        try:
            lik[k] = ast.literal_eval(v)             # last occurrence wins, as in MP
        except (ValueError, SyntaxError):
            lik[k] = v
    return pars, cosmo_args, cosmo_path, exp[0], lik


# ---------------------------------------------------------------------------
# theory code: the classy.py of the run's pipeline, loaded by path
# ---------------------------------------------------------------------------
_CLASSY = {}


def load_classy(cosmo_path):
    """Import <pipeline>/class/python/classy.py under a unique module name, so that
    runs from different pipelines can be used in the same kernel (if their
    dependencies are importable in it)."""
    py = to_local(cosmo_path) / "python" / "classy.py"
    key = str(py)
    if key not in _CLASSY:
        if not py.exists():
            raise FileNotFoundError(py)
        name = "classy_" + re.sub(r"\W+", "_", py.parent.parent.parent.name)
        spec = importlib.util.spec_from_file_location(name, py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _CLASSY[key] = mod
    return _CLASSY[key]


# ---------------------------------------------------------------------------
# a MontePython run
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, folder, root=None, burn_in=0.3, verbose=True):
        """folder: MP output folder (relative to omega_m/ or absolute).
        root: chain root inside it (e.g. '2026-10-22_100000_'); default: the newest
        root with at least one non-empty chain file (folders can hold several runs)."""
        from getdist import loadMCSamples

        self.folder = to_local(folder)
        self.name = self.folder.name
        (self.logp, self.cosmo_args, self.cosmo_path,
         self.experiment, self.lik) = read_log_param(self.folder)

        roots = {}
        for f in self.folder.glob("*.paramnames"):
            r = str(f)[:-len(".paramnames")]
            n = [c for c in glob.glob(glob.escape(r) + "_[0-9]*.txt") if Path(c).stat().st_size > 0]
            if n:
                roots[Path(r).name] = len(n)
        if not roots:
            raise FileNotFoundError(f"no chains in {self.folder}")
        if root is None:
            root = max(roots)                        # names start with the date -> newest
        elif root not in roots:
            raise ValueError(f"root {root!r} not in {sorted(roots)}")
        self.root = root
        if verbose:
            others = ", ".join(f"{r} ({n})" for r, n in sorted(roots.items()) if r != root)
            print(f"{self.name}: root {root} ({roots[root]} chains)"
                  + (f"; also in folder: {others}" if others else ""))

        s = loadMCSamples(str(self.folder / root), settings={"ignore_rows": burn_in}, no_cache=True)
        self.names = [p.name for p in s.paramNames.names]
        self.samples, self.weights, self.loglikes = s.samples, s.weights, s.loglikes

    # classy inputs ------------------------------------------------------------
    def _inputs(self, row):
        """classy input dict from one sample row: varied params from the row, fixed
        ones from log.param, both times 'scale'; plus data.cosmo_arguments."""
        p = dict(self.cosmo_args)
        for name, (mean, lo, hi, sigma, scale, role) in self.logp.items():
            if role != "cosmo":
                continue
            val = row[self.names.index(name)] if name in self.names else mean
            p[name] = val * scale
        return p

    def bestfit(self):
        i = int(np.argmin(self.loglikes))
        return self._inputs(self.samples[i])

    def draws(self, n, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.weights), size=n, replace=True, p=self.weights / self.weights.sum())
        return [self._inputs(self.samples[i]) for i in idx]

    @property
    def l_switch(self):
        """Fixed bin edges (empty for one bin)."""
        ls = sorted((int(k[8:]), v[0]) for k, v in self.logp.items()
                    if k.startswith("l_switch") and v[3] == 0)
        return [v for _, v in ls][:-1]

    @property
    def fiducial_file(self):
        return to_local(self.lik["fiducial_file"]) if "fiducial_file" in self.lik else None


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def spectra(run, pars):
    """{'ell', 'tt', 'ee', 'te' [muK^2], 'dd' = l(l+1) C_l^pp} for l = l_min..l_max of
    the run's likelihood, computed by the run's pipeline (as MontePython does)."""
    classy = load_classy(run.cosmo_path)
    lmin, lmax = int(run.lik["l_min"]), int(run.lik["l_max"])
    c = classy.Class()
    c.set(pars)
    c.compute()
    cl = c.lensed_cl(lmax)
    t2 = (c.T_cmb() * 1e6) ** 2
    ell = np.arange(lmin, lmax + 1)
    out = {"ell": ell,
           "tt": cl["tt"][ell] * t2, "ee": cl["ee"][ell] * t2, "te": cl["te"][ell] * t2,
           "dd": ell * (ell + 1.0) * cl["pp"][ell]}
    c.struct_cleanup()
    c.empty()
    return out


# ---------------------------------------------------------------------------
# mock
# ---------------------------------------------------------------------------
class Mock:
    """MontePython mock-CMB fiducial file: l, TT+N_T, EE+N_P, TE, DD+N_DD [, TD],
    with the noise of the run's likelihood (noise_from_file)."""

    def __init__(self, path, run):
        self.path = to_local(path)
        self.name = self.path.stem
        self.header = [l for l in self.path.read_text().splitlines() if l.startswith("#")]
        lik = run.lik
        if not lik.get("noise_from_file", False) or not lik.get("LensingExtraction", False):
            raise NotImplementedError("only noise_from_file + LensingExtraction (fake_planck_realistic)")
        if lik.get("Bmodes", False) or lik.get("ExcludeTTTEEE", False):
            raise NotImplementedError("Bmodes / ExcludeTTTEEE")
        lmin, lmax = int(lik["l_min"]), int(lik["l_max"])
        self.ell = np.arange(lmin, lmax + 1)
        self.f_sky = float(lik["f_sky"])

        d = np.loadtxt(self.path, comments="#")
        d = d[(d[:, 0] >= lmin) & (d[:, 0] <= lmax)]
        if not np.array_equal(d[:, 0].astype(int), self.ell):
            raise ValueError(f"{self.path}: l column does not cover {lmin}..{lmax}")
        obs = {"tt": d[:, 1], "ee": d[:, 2], "te": d[:, 3], "dd": d[:, 4]}   # incl. noise

        ddir = to_local(lik["data_directory"])
        n = np.loadtxt(ddir / lik["noise_file"], comments="#")
        n = n[(n[:, 0] >= lmin) & (n[:, 0] <= lmax)]
        if not np.array_equal(n[:, 0].astype(int), self.ell):
            raise ValueError("noise file l column mismatch")
        l = self.ell
        self.noise = {"tt": n[:, 1], "ee": n[:, 2], "te": np.zeros(len(l)),
                      "dd": n[:, 3] / (l * (l + 1) / 2 / np.pi)}           # as in MP
        self.obs = obs
        self.signal = {k: obs[k] - self.noise[k] for k in SPECTRA}
        # Gaussian per-l error of each spectrum (cosmic variance + noise, f_sky),
        # diagonal: ignores the TT-TE-EE correlations the likelihood has.
        nu = (2 * l + 1) * self.f_sky
        self.sigma = {"tt": np.sqrt(2 / nu) * obs["tt"],
                      "ee": np.sqrt(2 / nu) * obs["ee"],
                      "te": np.sqrt((obs["te"] ** 2 + obs["tt"] * obs["ee"]) / nu),
                      "dd": np.sqrt(2 / nu) * obs["dd"]}


# ---------------------------------------------------------------------------
# residuals
# ---------------------------------------------------------------------------
KINDS = {   # residual = f_l (C_model - C_mock); name: y-label (f_l in _factor)
    "sigma": "$\\Delta C_\\ell / \\sigma_\\ell$",
    "frac":  "$\\Delta C_\\ell / C_\\ell$",
    "dl":    "$\\Delta D_\\ell$",
}


def _factor(kind, spec, mock):
    l = mock.ell
    if kind == "sigma":
        return 1.0 / mock.sigma[spec]
    if kind == "frac":
        if spec == "te":   # TE crosses zero: normalise by sqrt(C_TT C_EE)
            return 1.0 / np.sqrt(mock.signal["tt"] * mock.signal["ee"])
        return 1.0 / mock.signal[spec]
    if kind == "dl":       # D_l = l(l+1) C_l / 2pi  (DD -> [l(l+1)]^2 C_pp / 2pi)
        return l * (l + 1) / (2 * np.pi)
    raise ValueError(f"kind must be one of {list(KINDS)}")


def bins(mock, width):
    """l-bin edges [lo, hi). width: None or 1 -> no binning; an int -> bins of that
    width; an array -> these edges (e.g. np.geomspace(2, 2509, 60) for a log axis;
    rounded to integers, duplicates and empty bins dropped)."""
    if width is None or np.isscalar(width):
        if not width or width <= 1:
            return None
        e = np.arange(mock.ell[0], mock.ell[-1] + 1, int(width))
        return np.append(e, mock.ell[-1] + 1)
    e = np.unique(np.round(np.asarray(width, dtype=float)).astype(int))
    e = e[(e >= mock.ell[0]) & (e <= mock.ell[-1] + 1)]
    if len(e) < 2:
        raise ValueError("need at least two bin edges inside the l range")
    return e


def residual(cl, spec, mock, kind="sigma", edges=None):
    """(l, r): residual of model spectrum cl[spec] w.r.t. the mock.
    Unbinned: r_l = f_l (C_l^model - C_l^mock).
    Binned (edges): inverse-variance weighted over each bin, w_l = 1/sigma_l^2;
      kind 'sigma' -> sum(w dC)/sqrt(sum w): the bandpower deviation in units of
                      the bandpower error;
      other kinds  -> weighted mean of r_l (weights 1/(f_l sigma_l)^2)."""
    dc = cl[spec] - mock.signal[spec]
    f = _factor(kind, spec, mock)
    if edges is None:
        return mock.ell, f * dc
    lc, r = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mock.ell >= lo) & (mock.ell < hi)
        w = 1.0 / mock.sigma[spec][m] ** 2
        lc.append(np.sum(w * mock.ell[m]) / np.sum(w))
        if kind == "sigma":
            r.append(np.sum(w * dc[m]) / np.sqrt(np.sum(w)))
        else:
            wr = 1.0 / (f[m] * mock.sigma[spec][m]) ** 2
            r.append(np.sum(wr * f[m] * dc[m]) / np.sum(wr))
    return np.array(lc), np.array(r)


def ylabel(kind, spec):
    s = {"tt": "TT", "te": "TE", "ee": "EE", "dd": "dd"}[spec]
    if kind == "frac" and spec == "te":
        return "$\\Delta C_\\ell^{TE} / \\sqrt{C_\\ell^{TT} C_\\ell^{EE}}$"
    if kind == "dl":
        return f"$\\Delta D_\\ell^{{{s}}}$" + ("" if spec == "dd" else r" [$\mu$K$^2$]")
    return KINDS[kind].replace("C_\\ell", f"C_\\ell^{{{s}}}")
