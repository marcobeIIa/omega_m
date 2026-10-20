"""dC_l/domega_m at fixed theta_s from the with_cp_lcdm pipeline (CosmoPower lcdm v1, Bolliet+2023).

Uses the pipeline's own with_cp_lcdm/class/python/classy.py, unmodified:
  v1_an        Class.dcl_domega_m(): exact forward-mode JVP through the networks along
               (d omega_cdm = 1, dH0 = 100 h/omega_m * s), s = the pipeline's fitted dlnh/dlnomega_m (CLASS fit)
  v1_an_exact  same, with s fixed to the exact CLASS value at the fiducial (from compute_derivs.py)
  v1_num       5-point stencil of lensed_cl() at the same (omega_m, h) points as CLASS (fixed theta_s)
Needs dcl_domegam_spt_vs_class.npz (stencil h values, eps). Without TensorFlow, put cpstub/ on PYTHONPATH
(numpy cosmopower stand-in); with the real cosmopower env it is not needed.
Run: PYTHONPATH=<with_cp_lcdm/class/python>[:cpstub] python compute_v1.py
"""
import os
import numpy as np
import classy

HERE = os.path.dirname(os.path.abspath(__file__))
assert hasattr(classy.Class, 'dcl_domega_m'), 'not the with_cp_lcdm stand-in: %s' % classy.__file__
z = np.load(os.path.join(HERE, 'dcl_domegam_spt_vs_class.npz'))
EPS = float(z['eps']); hk = {int(k): h for k, h in z['h_points']}
L = 2508; T2 = 2.7255e6**2
OMB, OMC = 0.02237, 0.12


def pars(k, **extra):
    p = {'omega_b': OMB, 'h': hk[k], 'tau_reio': 0.054, 'n_s': 0.965, 'ln10^{10}A_s': 3.044,
         'omega_m0': OMB + OMC + k * EPS + classy.OMEGA_NU, 'l_switch1': L}
    p.update(extra)
    return p


def cls(p):
    c = classy.Class(); c.set(p); c.compute(); r = c.lensed_cl(L)
    return np.array([r['tt'], r['te'], r['ee']]) * T2


def dcl(p):
    c = classy.Class(); c.set(p); c.compute(); r = c.dcl_domega_m(L)
    return np.array([r['tt'], r['te'], r['ee']]) * T2


poly = {'dwm_b': 0.0, 'l_c': 400.0}                       # switches the [poly] derivative on, adds nothing
om = OMB + OMC + classy.OMEGA_NU
s_pipe = classy._dlnh_dlnomega_m(OMB, om, hk[0])
s_exact = float(z['slope']) * om / hk[0]
v1_an = dcl(pars(0, **poly))
v1_an_exact = dcl(pars(0, dlnh_dlnomega_m=s_exact, **poly))
C = {k: cls(pars(k)) for k in (-2, -1, 1, 2, -4, 4)}
v1_num = (C[-2] - 8 * C[-1] + 8 * C[1] - C[2]) / (12 * EPS)
v1_num_step = (C[-4] - 8 * C[-2] + 8 * C[2] - C[4]) / (24 * EPS)
C0 = cls(pars(0))
np.savez(os.path.join(HERE, 'dcl_domegam_lcdm_v1.npz'), ell=np.arange(L + 1), C_v1=C0, v1_an=v1_an,
         v1_an_exact=v1_an_exact, v1_num=v1_num, v1_num_step=v1_num_step, s_pipe=s_pipe, s_exact=s_exact,
         classy_file=classy.__file__)
print('dlnh/dlnomega_m: pipeline fit %.5f, exact CLASS %.5f' % (s_pipe, s_exact))
