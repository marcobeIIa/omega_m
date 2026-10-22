"""
check_install.py -- is this mAxiCLASS build giving the right cosmologies?

Run from omega_m/ in the env you built it with (conda activate maxiclass):
    python with_maxiclass/check_install.py

1. git: branch/upstream/commit of with_maxiclass/maxiclass; must contain the h/H0
   fix 1de71f2 (mAxiCLASShoot; reference commit b5a3af9 used for the mocks in model/).
2. LCDM limit (no EDE) at omega_m 0.15 vs with_class_mnu/model/lcdm.dat (CLASS 3.4.1).
   Reference build: sqrt(chi2) = 0.05.
3. 2-axion EDE (make_mocks.sh recipe) vs with_maxiclass/model/ede2ax_pdh0_mean.dat.
   Reference build: identical (0.00); different compilers/OS: expect < ~0.2
   (the EDE shooting tolerance alone moves a mock by ~0.15).
sqrt(chi2) = Gaussian chi2 of the mock difference with the fake_planck_realistic
noise (TT TE EE dd, f_sky 0.57, l >= 30). Temporary mocks are deleted afterwards.
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # omega_m/
MAXI = os.path.abspath(os.environ.get("MAXICLASS_DIR",
                                      os.path.join(ROOT, "with_maxiclass", "maxiclass")))
MODEL = os.path.join(ROOT, "with_maxiclass", "model")
REF_COMMIT = "b5a3af9"
COS = ["--omega_b", "0.02267", "--h", "0.72645", "--tau_reio", "0.060",
       "--n_s", "0.988", "--lnAs", "3.067"]
EDE = ["N_mscf=2", "n_axion_mscf=3,3", "theta_ini_mscf=2.8,2.8",
       "fraction_maxion_ac=0.085,0.073", "log10_maxion_ac=-3.440,-4.000"]
results = []


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", MAXI] + list(args),
                                       stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return "?? (%s)" % e.__class__.__name__


def sqrt_chi2(fa, fb, lmin=30, fsky=0.57):
    A, B = np.loadtxt(fa), np.loadtxt(fb)
    l = A[:, 0]
    m = l >= lmin

    def mat(X):
        C = np.zeros((len(X), 3, 3))
        C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = X[:, 1], X[:, 2], X[:, 4]
        C[:, 0, 1] = C[:, 1, 0] = X[:, 3]
        return C
    M = np.linalg.solve(mat(A), mat(B) - mat(A))
    c2 = (2 * l + 1) * fsky / 2 * np.einsum("lij,lji->l", M, M)
    tt = np.abs(B[m, 1] / A[m, 1] - 1).max()
    return float(np.sqrt(c2[m].sum())), float(tt)


def make(name, args):
    cmd = [sys.executable, os.path.join(ROOT, "models", "make_fiducial.py"),
           "--pipeline", "with_maxiclass", "--l_switch", "2508", "--name", name,
           "--force"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("make_fiducial.py failed for %s" % name)
    return os.path.join(MODEL, name)


def report(label, ref, new, limit):
    s, tt = sqrt_chi2(ref, new)
    ok = s < limit
    results.append(ok)
    print("%-38s sqrt(chi2) = %.3f   max|dTT/TT| = %.2e   %s"
          % (label, s, tt, "OK" if ok else "<-- CHECK (limit %.2f)" % limit))


print("mAxiCLASS folder:", MAXI)
branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "--short", "HEAD")
upstream = git("rev-parse", "--abbrev-ref", "@{u}")
print("branch: %s (tracks %s)   commit: %s   (reference: mAxiCLASShoot %s)"
      % (branch, upstream, commit, REF_COMMIT))
# what matters: the h/H0 reading fix 1de71f2 (missing on MP-branch) is in this checkout
has_h_fix = subprocess.run(["git", "-C", MAXI, "merge-base", "--is-ancestor", "1de71f2", "HEAD"]).returncode == 0
print("contains the h/H0 fix (1de71f2):", "yes" if has_h_fix else "NO")
print("last commit:", git("log", "-1", "--format=%cd  %s", "--date=short"))
dirty = git("status", "--porcelain", "--untracked-files=no")
print("local changes to tracked files:", "none" if not dirty else "\n" + dirty)
results.append(has_h_fix)
if not has_h_fix:
    print("  <-- 'h' is not read by this code (MP-branch?): switch to mAxiCLASShoot, then rebuild")

sys.path.insert(0, os.path.join(ROOT, "with_maxiclass", "class", "python"))
import classy  # noqa: E402  (the with_maxiclass wrapper)
print("compiled module:", classy.REAL_CLASSY_FILE)
print()

tmp = []
try:
    f = make("_check_lcdm.dat", ["--omega_m", "0.15"]); tmp.append(f)
    report("LCDM limit vs with_class_mnu/lcdm.dat", os.path.join(ROOT, "with_class_mnu", "model", "lcdm.dat"), f, 0.2)
    f = make("_check_ede.dat", ["--omega_m", "0.15631"] + COS + ["--set"] + EDE); tmp.append(f)
    print("  ", open(f).readlines()[2].strip())
    report("2-axion EDE vs ede2ax_pdh0_mean.dat", os.path.join(MODEL, "ede2ax_pdh0_mean.dat"), f, 0.3)
finally:
    for f in tmp:
        if os.path.exists(f):
            os.remove(f)

print("\nALL OK" if all(results) else "\nSOMETHING TO CHECK (see above)")
