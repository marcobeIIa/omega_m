"""Shared loading for the EDE-mock slides (deck 3). Chains: with_cp_lcdm/chains, newest run in each folder.
Same conventions as analyse_poly.ipynb / analyse_chain.ipynb: burn-in 0.3, omega_b rescaled by its log.param scale."""
import ast, glob, os, re
from pathlib import Path
import numpy as np
from getdist import loadMCSamples

U = Path(os.environ.get("OMEGA_M_ROOT", Path(__file__).resolve().parents[2]))
CH = U / "with_cp_lcdm" / "chains"
BURN_IN = 0.3
EDE2 = "model_ede2ax_pdh0_mean_from_maxiclass__param_"
EDE1 = "model_ede1ax_f012_z1e4_from_maxiclass__param_"
MOCK = {EDE2: U / "with_maxiclass/model/ede2ax_pdh0_mean.dat", EDE1: U / "with_maxiclass/model/ede1ax_f012_z1e4.dat"}


def log_param(folder):
    text = (folder / "log.param").read_text()
    return {n: ast.literal_eval(r) for n, r in
            re.findall(r"^data\.parameters\[['\"](.+?)['\"]\]\s*=\s*(\[.*?\])", text, re.M)}


def newest_root(folder):
    roots = {}
    for f in folder.glob("*.paramnames"):
        root = str(f)[:-len(".paramnames")]
        ch = [c for c in glob.glob(glob.escape(root) + "_[0-9]*.txt") if Path(c).stat().st_size > 0]
        if ch:
            roots[root] = max(Path(c).stat().st_mtime for c in ch)
    return max(roots, key=roots.get)


def load(run):
    folder = CH / run
    logp = log_param(folder)
    root = newest_root(folder)
    s = loadMCSamples(root, settings={"ignore_rows": BURN_IN}, no_cache=True)
    for i, n in enumerate(s.getParamNames().list()):
        if n in logp and logp[n][4] not in (1, 1.0):
            s.samples[:, i] *= logp[n][4]
    s.updateBaseStatistics()
    return s, logp, root


def chain_info(root):
    files = [f for f in glob.glob(glob.escape(root) + "_[0-9]*.txt") if Path(f).stat().st_size > 0]
    raw = [np.atleast_2d(np.loadtxt(f)) for f in files]
    steps = [int(c[:, 0].sum()) for c in raw]
    acc = sum(len(c) for c in raw) / sum(steps)
    chi2 = 2 * min(c[:, 1].min() for c in raw)
    return raw, steps, acc, chi2


def max_gr(raw, ncols):
    """Max over parameters of the weighted Gelman-Rubin R-1 (as in analyse_chain.ipynb), after burn-in."""
    out = []
    for col in range(2, 2 + ncols):
        means, vars_, ns = [], [], []
        for c in raw:
            c = c[int(BURN_IN * len(c)):]
            w, x = c[:, 0], c[:, col]
            m = np.average(x, weights=w); means.append(m)
            vars_.append(np.average((x - m) ** 2, weights=w)); ns.append(w.sum())
        W = np.mean(vars_); B = np.var(means, ddof=1); n = np.mean(ns)
        out.append(((n - 1) / n * W + B) / W - 1)
    return max(out)


def mock_header(run_prefix):
    h = MOCK[run_prefix].open().readline()
    return {k: float(v) for k, v in re.findall(r"([\w^{}]+)\s*=\s*([-\d.eE+]+)", h)}


def bins(logp):
    sw = sorted((int(k[len("l_switch"):]), v[0]) for k, v in logp.items() if k.startswith("l_switch"))
    edges = [2] + [int(v) for _, v in sw]
    return [(edges[i] if i == 0 else edges[i] + 1, edges[i + 1]) for i in range(len(edges) - 1)]
