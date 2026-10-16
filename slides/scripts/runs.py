"""Load every MontePython run used in the deck (same conventions as analyse_chain.ipynb)."""
import re, ast, glob
from pathlib import Path
import numpy as np
from getdist import loadMCSamples

U = Path("/mnt/user-data/uploads/omega_m")
BURN_IN = 0.3
LABELS = {"omega_b": r"\omega_b", "h": "h", "tau_reio": r"\tau_\mathrm{reio}", "n_s": "n_s",
          "ln10^{10}A_s": r"\ln(10^{10}A_s)", "H0": r"H_0"}

def read_log_param(d):
    text = (Path(d) / "log.param").read_text()
    P = {}
    for name, row in re.findall(r"^data\.parameters\[['\"](.+?)['\"]\]\s*=\s*(\[.*?\])", text, re.M):
        mean, lo, hi, sig, scale, role = ast.literal_eval(row)
        P[name] = dict(mean=mean, min=lo, max=hi, sigma=sig, scale=scale, role=role)
    fid = re.findall(r"^\w+\.fiducial_file\s*=\s*['\"](.+?)['\"]", text, re.M)
    return P, fid[-1]

def read_fid(path):
    p = Path(path)
    parts = p.parts
    i = [k for k, s in enumerate(parts) if s.startswith("with_")][0]
    local = U.joinpath(*parts[i:])
    hdr = local.open().readline()
    out = {}
    for k, v in re.findall(r"([\w^{}]+)\s*=\s*([^,]+)", hdr):
        try: out[k] = float(v)
        except ValueError: pass
    return out

def bin_edges(P):
    sw = sorted((int(k[8:]), v["mean"]) for k, v in P.items() if k.startswith("l_switch"))
    e = [2] + [int(v) for _, v in sw]
    return [(e[i] if i == 0 else e[i] + 1, e[i + 1]) for i in range(len(e) - 1)]

def fid_steps(F):
    sw = sorted((int(k[8:]), v) for k, v in F.items() if k.startswith("l_switch"))
    om = sorted((int(k[7:]), v) for k, v in F.items() if re.fullmatch(r"omega_m\d+", k))
    e = [2] + [int(v) for _, v in sw]
    return [(e[i] if i == 0 else e[i] + 1, e[i + 1], om[i][1]) for i in range(len(om))]

def load(d, label=None):
    d = Path(d)
    root = str(sorted(d.glob("*.paramnames"))[0])[:-len(".paramnames")]
    P, fidf = read_log_param(d)
    s = loadMCSamples(root, settings={"ignore_rows": BURN_IN}, no_cache=True)
    names = s.getParamNames().list()
    for i, n in enumerate(names):
        sc = P.get(n, {}).get("scale", 1)
        if sc != 1: s.samples[:, i] *= sc
    s.updateBaseStatistics()
    rng = {}
    for n in names:
        if n in P and P[n]["role"] != "derived":
            sc = P[n]["scale"]; lo, hi = P[n]["min"], P[n]["max"]
            rng[n] = (None if lo in (None, -1) else lo * sc, None if hi in (None, -1) else hi * sc)
    s.setRanges(rng)
    for p in s.paramNames.names:
        if p.name in LABELS: p.label = LABELS[p.name]
        m = re.fullmatch(r"omega_m(\d+)", p.name)
        if m: p.label = rf"\omega_{{m,{m.group(1)}}}"
    oms = sorted([n for n in names if re.fullmatch(r"omega_m\d+", n)], key=lambda n: int(n[7:]))
    col = lambda i: s.samples[:, s.index[f"omega_m{i}"]].copy()
    for i in range(1, len(oms)):
        s.addDerived(col(i) - col(0), name=f"domega_m{i}", label=rf"\omega_{{m,{i}}}-\omega_{{m,0}}")
    for i in range(2, len(oms)):
        for j in range(1, i):
            s.addDerived(col(i) - col(j), name=f"domega_m{i}_{j}", label=rf"\omega_{{m,{i}}}-\omega_{{m,{j}}}")
    s.updateBaseStatistics()
    # raw chains (all rows) for chi2_min, acceptance, R-1
    files = sorted(glob.glob(glob.escape(root) + "_[0-9]*.txt"))
    raw = [np.atleast_2d(np.loadtxt(f)) for f in files]
    s.raw, s.P, s.fid = raw, P, read_fid(fidf)
    s.bins, s.steps, s.oms = bin_edges(P), fid_steps(s.fid), oms
    s.name_tag = label or d.name
    s.chi2_min = 2 * min(c[:, 1].min() for c in raw)
    s.acc = np.mean([len(c) / c[:, 0].sum() for c in raw])
    return s

def run_dir(model, param, pipeline="with_cosmopower"):
    return U / pipeline / "chains" / f"model_{model}__param_{param}"
