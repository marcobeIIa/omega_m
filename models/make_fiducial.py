"""
make_fiducial.py -- write a fiducial ("fake data") file for MontePython's
fake_planck_realistic likelihood, with either pipeline:

    USE_CLASS = False  ->  CosmoPower stand-in  (with_cosmopower/class/python)
                           written to with_cosmopower/model/<name>
    USE_CLASS = True   ->  real CLASS, binned   (with_class/class/python)
                           written to with_class/model/<name>

Set USE_CLASS below, or override it on the command line with --use_class /
--use_cosmopower. Any other pipeline folder (e.g. with_class_mnu: real CLASS
with one 0.06 eV neutrino + HMcode) is chosen with --pipeline <folder>.
All pipelines use the same binned-omega_m model, so the same arguments give
the same cosmology in each (with_class_mnu / with_cp_lcdm: omega_m includes
the neutrino density).

The file has exactly the format MontePython writes itself on a first run:
    l   C_l^TT+N_T   C_l^EE+N_P   C_l^TE   l(l+1)C_l^pp+N_dd   C_l^Td(=0)
in muK^2, for l = 2 .. l_max, with the same Planck noise added.

Example (null test, both bins at 0.15), from omega_m/:
    python models/make_fiducial.py --omega_m 0.15 0.15 --l_switch 1250 2508 \
        --name fid_two_bins_null.dat               # uses USE_CLASS below
    python models/make_fiducial.py --use_class --omega_m 0.15 0.15 \
        --l_switch 1250 2508 --name fid_two_bins_null.dat

Example (CLASS mock with massive neutrino, for the with_cp_lcdm emulator):
    python models/make_fiducial.py --pipeline with_class_mnu --omega_m 0.143 \
        --l_switch 2508 --name lcdm.dat

Example (injection):
    python models/make_fiducial.py --omega_m 0.12 0.17 --l_switch 1250 2508 \
        --name fid_two_bins_012_017.dat

Other parameters default to the means in two_bins.param; change them with
--omega_b (the real value, e.g. 0.0224) --h --tau_reio --n_s --lnAs.
"""

import argparse
import math
import os
import sys

# ---- switch -----------------------------------------------------------------
USE_CLASS = False   # True: real CLASS (with_class/), False: CosmoPower (with_cosmopower/)
# ----------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # omega_m/
NOISE_FILE = os.path.join(ROOT, "montepython_fake", "data",
                          "noise_fake_planck_realistic_two.dat")


def read_noise(path, l_min, l_max):
    """Same parsing as MontePython's Likelihood_mock_cmb (noise_from_file)."""
    nT, nP, ndd = {}, {}, {}
    with open(path) as f:
        for line in f:
            if "#" in line or not line.strip():
                continue
            c = line.split()
            l = int(float(c[0]))
            if l_min <= l <= l_max:
                nT[l] = float(c[1])
                nP[l] = float(c[2])
                ndd[l] = float(c[3]) / (l * (l + 1) / 2. / math.pi)
    missing = [l for l in range(l_min, l_max + 1) if l not in nT]
    if missing:
        raise ValueError("noise file has no entry for l = %s ..." % missing[:5])
    return nT, nP, ndd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omega_m", type=float, nargs="+", required=True,
                    help="omega_m in each bin, e.g. 0.15 0.15")
    ap.add_argument("--l_switch", type=float, nargs="+", required=True,
                    help="upper edge of each bin, last = 2508, e.g. 1250 2508")
    ap.add_argument("--name", required=True,
                    help="file name only, e.g. fid_two_bins_null.dat; it goes "
                         "into with_class/model/ or with_cosmopower/model/")
    sw = ap.add_mutually_exclusive_group()
    sw.add_argument("--use_class", dest="use_class", action="store_true", default=None,
                    help="real CLASS (overrides USE_CLASS)")
    sw.add_argument("--use_cosmopower", dest="use_class", action="store_false",
                    help="CosmoPower (overrides USE_CLASS)")
    sw.add_argument("--pipeline", default=None,
                    help="any pipeline folder in omega_m/ with class/python/classy.py, "
                         "e.g. with_class_mnu or with_cp_lcdm (overrides USE_CLASS)")
    ap.add_argument("--omega_b", type=float, default=0.0224)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--tau_reio", type=float, default=0.054)
    ap.add_argument("--n_s", type=float, default=0.965)
    ap.add_argument("--lnAs", type=float, default=3.044, help="ln10^{10}A_s")
    ap.add_argument("--l_min", type=int, default=2)
    ap.add_argument("--l_max", type=int, default=2508)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    a = ap.parse_args()

    if a.pipeline is not None:
        pipeline = a.pipeline.strip("/")
    else:
        use_class = USE_CLASS if a.use_class is None else a.use_class
        pipeline = "with_class" if use_class else "with_cosmopower"
    if os.path.basename(a.name) != a.name:
        sys.exit("--name is a file name, not a path: the folder is chosen by the switch")
    a.out = os.path.join(ROOT, pipeline, "model", a.name)

    wrapper_dir = os.path.join(ROOT, pipeline, "class", "python")
    if not os.path.isfile(os.path.join(wrapper_dir, "classy.py")):
        sys.exit("no classy.py in %s" % wrapper_dir)
    sys.path.insert(0, wrapper_dir)
    import classy
    from classy import Class
    print("pipeline: %s  (classy wrapper: %s)" % (pipeline, classy.__file__))
    if hasattr(classy, "REAL_CLASSY_FILE"):
        print("real CLASS: %s" % classy.REAL_CLASSY_FILE)

    if len(a.omega_m) != len(a.l_switch):
        sys.exit("need one --omega_m per bin: got %d omega_m and %d l_switch"
                 % (len(a.omega_m), len(a.l_switch)))
    if os.path.exists(a.out) and not a.force:
        sys.exit("%s already exists (use --force to overwrite)" % a.out)

    params = {"omega_b": a.omega_b, "h": a.h, "tau_reio": a.tau_reio,
              "n_s": a.n_s, "ln10^{10}A_s": a.lnAs, "l_max_scalars": a.l_max}
    for i, (om, ls) in enumerate(zip(a.omega_m, a.l_switch)):
        params["omega_m%d" % i] = om
        params["l_switch%d" % (i + 1)] = ls

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    cl = cosmo.lensed_cl(a.l_max)
    T = cosmo.T_cmb() * 1e6  # same conversion as MontePython's get_cl
    tt, ee, te = cl["tt"] * T**2, cl["ee"] * T**2, cl["te"] * T**2
    pp, tp = cl["pp"], cl["tp"] * T

    nT, nP, ndd = read_noise(NOISE_FILE, a.l_min, a.l_max)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# Fiducial parameters, pipeline = %s" % pipeline)
        for k, v in params.items():
            f.write(", %s = %.5g" % (k, v))
        f.write("\n")
        for l in range(a.l_min, a.l_max + 1):
            f.write("%5d  " % l)
            f.write("%.8g  " % (tt[l] + nT[l]))
            f.write("%.8g  " % (ee[l] + nP[l]))
            f.write("%.8g  " % te[l])
            f.write("%.8g  " % (l * (l + 1.) * pp[l] + ndd[l]))
            f.write("%.8g  " % (math.sqrt(l * (l + 1.)) * tp[l]))
            f.write("\n")
    print("wrote %s  (l = %d..%d, omega_m = %s, l_switch = %s)"
          % (a.out, a.l_min, a.l_max, a.omega_m, a.l_switch))


if __name__ == "__main__":
    main()
