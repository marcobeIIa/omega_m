"""
run_mcmc.py -- launch / monitor / stop MontePython chains for the omega_m project.

Use it from a notebook (no terminal needed):

    import run_mcmc as rm
    rm.launch("two_bins.param", n_chains=4, N=20000)   # returns immediately
    rm.status("two_bins.param")                         # progress of each chain
    rm.stop("two_bins.param")                           # kill the chains

or from a terminal:

    python run_mcmc.py launch two_bins.param --chains 4 -N 20000
    python run_mcmc.py status two_bins.param
    python run_mcmc.py stop   two_bins.param

What launch() does:
  1. Checks that every likelihood's l_max is <= 2508 (where the emulators stop).
  2. If the mock-likelihood fiducial file doesn't exist yet, runs MontePython
     once and waits. That run writes the fiducial spectra (from the param file's
     mean values) and exits; MontePython always stops after writing a fiducial.
  3. Starts n_chains independent chains in the background, writing to
     chains/<param name>/. They are detached, so closing the notebook or the
     terminal does not kill them. Each chain logs to chains/<name>/logs/.

Chains run with the same Python as this script (in a notebook: the kernel's
Python). Run it from the environment that has cosmopower + tensorflow.
"""

import argparse
import ast
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/bellamarco01/uni/omega_m")
MP_DIR = ROOT / "montepython_fake"
MP_EXE = MP_DIR / "montepython" / "MontePython.py"
MP_CONF = MP_DIR / "default.conf"
MP_DATA = MP_DIR / "data"
CHAINS = ROOT / "chains"
SHIM_DIR = ROOT / "class" / "python"   # the CosmoPower-backed classy.py
L_MAX_EMU = 2508
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _param_path(param):
    p = Path(param)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _out_dir(param):
    return CHAINS / _param_path(param).stem


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SHIM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    # Each chain evaluates a tiny batch: one CPU thread per chain is enough,
    # and stops parallel chains from fighting over cores.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return env


def _setting(text, name):
    """Last uncommented `name = value` in a param/.data file, or None."""
    val = None
    for line in text.splitlines():
        if "#" in line:  # MontePython ignores any line containing '#'
            continue
        m = re.match(r"\s*%s\s*=\s*(.+)$" % re.escape(name), line)
        if m:
            val = m.group(1).strip()
    return val


def _likelihood_settings(param_file):
    """{experiment: {'l_max': int, 'fiducial_file': str|None}}, param file overrides .data."""
    ptext = param_file.read_text()
    exps = _setting(ptext, "data.experiments")
    if exps is None:
        raise ValueError("no data.experiments line in %s" % param_file)
    out = {}
    for exp in ast.literal_eval(exps):
        dfile = MP_DIR / "montepython" / "likelihoods" / exp / (exp + ".data")
        dtext = dfile.read_text() if dfile.exists() else ""
        s = {}
        for key in ("l_max", "fiducial_file"):
            v = _setting(ptext, "%s.%s" % (exp, key))
            if v is None:
                v = _setting(dtext, "%s.%s" % (exp, key))
            s[key] = v
        if s["l_max"] is not None:
            s["l_max"] = int(float(s["l_max"]))
        if s["fiducial_file"] is not None:
            s["fiducial_file"] = ast.literal_eval(s["fiducial_file"])
        out[exp] = s
    return out


def _mp_cmd(param, out, N, chain_number, covmat=None, extra=()):
    cmd = [PYTHON, str(MP_EXE), "run",
           "-p", str(param), "-o", str(out), "-N", str(N),
           "--conf", str(MP_CONF), "--chain-number", str(chain_number)]
    if covmat:
        cmd += ["-c", str(covmat)]
    return cmd + list(extra)


def _pidfile(out):
    return out / "logs" / "pids.txt"


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def launch(param, n_chains=4, N=20000, covmat=None, extra=()):
    param = _param_path(param)
    out = _out_dir(param)
    logs = out / "logs"

    if (out / "log.param").exists():
        print("NOTE: %s already has a log.param. MontePython will use THAT, not the "
              "current %s.\n      Use a new folder if you changed the param file."
              % (out, param.name))

    # 1. sanity checks on the likelihood settings
    settings = _likelihood_settings(param)
    for exp, s in settings.items():
        if s["l_max"] is not None and s["l_max"] > L_MAX_EMU:
            raise ValueError(
                "%s.l_max = %d, but the emulators stop at %d. Add\n    %s.l_max = %d\n"
                "to %s." % (exp, s["l_max"], L_MAX_EMU, exp, L_MAX_EMU, param.name))

    logs.mkdir(parents=True, exist_ok=True)

    # 2. fiducial file (mock likelihoods): create it first if missing
    missing = [MP_DATA / s["fiducial_file"] for s in settings.values()
               if s["fiducial_file"] and not (MP_DATA / s["fiducial_file"]).exists()]
    if missing:
        print("Creating fiducial file(s): %s" % ", ".join(m.name for m in missing))
        log = logs / "fiducial.log"
        with open(log, "w") as f:
            subprocess.run(_mp_cmd(param, out, 1, 0, covmat, extra), cwd=MP_DIR,
                           env=_env(), stdout=f, stderr=subprocess.STDOUT)
        still = [m for m in missing if not m.exists()]
        if still:
            raise RuntimeError("fiducial not written -- see %s" % log)
        # that run stops before sampling; drop its empty chain file
        for f in out.glob("*__0.txt"):
            if f.stat().st_size == 0:
                f.unlink()
        print("  done (log: %s)" % log)

    # 3. the chains, detached, each with its own chain number
    used = [int(m.group(1)) for f in out.glob("*.txt")
            for m in [re.search(r"__(\d+)\.txt$", f.name)] if m]
    first = max(used + [0]) + 1
    pids = []
    for k in range(n_chains):
        num = first + k
        log = open(logs / ("chain_%d.log" % num), "w")
        p = subprocess.Popen(_mp_cmd(param, out, N, num, covmat, extra), cwd=MP_DIR,
                             env=_env(), stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        pids.append((num, p.pid))
        if k == 0 and not (out / "log.param").exists():
            # let the first chain write log.param before the others read it
            for _ in range(120):
                if (out / "log.param").exists():
                    break
                time.sleep(1)
        else:
            time.sleep(2)

    with open(_pidfile(out), "a") as f:
        for num, pid in pids:
            f.write("%d %d\n" % (num, pid))
    print("Started %d chain(s) in %s (chain numbers %s)."
          % (n_chains, out, ", ".join(str(n) for n, _ in pids)))
    print("Check progress with status('%s')." % param.name)


def status(param):
    out = _out_dir(param)
    pf = _pidfile(out)
    if not pf.exists():
        print("No chains launched from here for %s." % out)
        return
    for line in pf.read_text().split("\n"):
        if not line.strip():
            continue
        num, pid = (int(x) for x in line.split())
        files = list(out.glob("*__%d.txt" % num))
        n_lines = 0
        for f in files:
            with open(f) as fh:
                n_lines = sum(1 for l in fh if l.strip() and not l.startswith("#"))
        log = out / "logs" / ("chain_%d.log" % num)
        last = ""
        if log.exists():
            tail = [l for l in log.read_text().splitlines() if l.strip()]
            last = tail[-1][:100] if tail else ""
        state = "running" if _alive(pid) else "stopped"
        print("chain %2d  pid %6d  %-8s  %6d accepted points   | %s"
              % (num, pid, state, n_lines, last))


def stop(param):
    out = _out_dir(param)
    pf = _pidfile(out)
    if not pf.exists():
        print("Nothing to stop.")
        return
    for line in pf.read_text().split("\n"):
        if line.strip():
            num, pid = (int(x) for x in line.split())
            if _alive(pid):
                os.killpg(pid, signal.SIGTERM)
                print("stopped chain %d (pid %d)" % (num, pid))
    pf.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["launch", "status", "stop"])
    ap.add_argument("param")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("-N", type=int, default=20000)
    ap.add_argument("-c", "--covmat", default=None)
    a = ap.parse_args()
    if a.action == "launch":
        launch(a.param, a.chains, a.N, a.covmat)
    elif a.action == "status":
        status(a.param)
    else:
        stop(a.param)
