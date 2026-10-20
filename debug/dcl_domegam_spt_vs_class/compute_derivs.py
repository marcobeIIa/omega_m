"""dC_l/domega_m at fixed 100*theta_s (and omega_b, tau, n_s, ln10^10 A_s): SPT emulator vs CLASS.

Three derivatives, l = 2..2508, TT TE EE (lensed, muK^2):
  an   SPT_high_accuracy LCDM emulator (Balkenhol+2022), ANALYTIC: exact network Jacobian (spt_emu.py)
         dC/domega_m|theta = dC/domch2 + dC/dh * (dh/domega_m)|theta,
         (dh/domega_m)|theta = -(dtheta/domega_cdm)/(dtheta/dh) from CLASS (5-point, fixed h, thermodynamics only).
  num  same emulator, NUMERICAL: 5-point stencil in omega_m, h(omega_m) solved at every stencil point so that
         CLASS 100*theta_s equals the fiducial value (brentq, xtol 1e-12).
  cls  CLASS 3.4.1, NUMERICAL: same 5-point stencil, same h(omega_m) points.
Physics (both): 1 massive nu 0.06 eV + N_ur 2.0328, HMcode lensing (what the SPT emulator was trained on).
omega_nu is fixed, so d/domega_m at fixed omega_b = d/domega_cdm.

CLASS precision. Default CLASS (and l_linstep 10 / accurate_lensing / delta_l_max 800 alone) is NOT enough: the
derivative jitters l-to-l at 10 < l < 150 by up to ~0.1 C_l per unit omega_m (transfer q-sampling), and l_max_scalars
= 2508 biases lensed C_l near the edge (TT -0.7% at l = 2500), so spectra are computed to 3500 and cut at 2508.
  BASE = l_max_scalars 3500, l_linstep 5, l_logstep 1.04, accurate_lensing 1, delta_l_max 1000
  Q    = q_linstep 0.08, q_logstep_spline 20, q_logstep_trapzd 5, hyper_sampling_flat 30, hyper_x_tol 1e-6
  TOL  = tol_perturbations_integration 1e-7, perturbations_sampling_stepsize 0.02, tol_ncdm_synchronous 1e-6,
         l_max_g 30, l_max_pol_g 30, l_max_ur 50, l_max_ncdm 50
  REF = BASE + Q + TOL (~9 min per run).  Checks written alongside:
  cls_step  REF, 5-point with step 2*EPS                     (step-size error = cls_step - cls)
  cls_low   BASE + Q (no TOL), 5-point step EPS               (precision error = cls_low - cls)
Output: dcl_domegam_spt_vs_class.npz.  ~35 min on 2 cores.
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import brentq
import classy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spt_emu

EMU_DIR = os.environ.get('SPT_EMU_DIR', os.path.join(HERE, '..', '..', 'with_cosmopower', 'cosmopower', 'cosmopower',
                                                   'trained_models', 'SPT_high_accuracy'))
L = 2508
TCMB = 2.7255e6
FID = dict(omega_b=0.02237, omega_cdm=0.12, h=0.674, tau_reio=0.054, n_s=0.965, ln10A=3.044)
PHYS = dict(N_ur=2.0328, N_ncdm=1, m_ncdm=0.06)
BASE = dict(l_max_scalars=3500, l_linstep=5, l_logstep=1.04, accurate_lensing=1, delta_l_max=1000)
Q = dict(q_linstep=0.08, q_logstep_spline=20., q_logstep_trapzd=5., hyper_sampling_flat=30., hyper_x_tol=1e-6)
TOL = dict(tol_perturbations_integration=1e-7, perturbations_sampling_stepsize=0.02, tol_ncdm_synchronous=1e-6,
           l_max_g=30, l_max_pol_g=30, l_max_ur=50, l_max_ncdm=50)
REF = dict(BASE, **Q, **TOL)
LOW = dict(BASE, **Q)
EPS = 1.2e-3                       # 5-point step in omega_m (~1 sigma); points at +-EPS, +-2 EPS (+-4 EPS for the check)
OFFS = [-4, -2, -1, 1, 2, 4]


def _base(dom, h):
    return dict(PHYS, omega_b=FID['omega_b'], omega_cdm=FID['omega_cdm'] + dom, h=h, tau_reio=FID['tau_reio'],
                n_s=FID['n_s'], A_s=np.exp(FID['ln10A']) * 1e-10)


def theta(dom, h):
    c = classy.Class(); c.set(_base(dom, h)); c.compute()
    t = c.get_current_derived_parameters(['100*theta_s'])['100*theta_s']
    c.struct_cleanup(); c.empty()
    return t


def spectra(args):
    dom, h, prec = args
    c = classy.Class(); s = _base(dom, h); s.update(output='tCl,pCl,lCl', lensing='yes', non_linear='hmcode'); s.update(prec)
    c.set(s); c.compute()
    r = c.lensed_cl(L); t = c.get_current_derived_parameters(['100*theta_s'])['100*theta_s']
    c.struct_cleanup(); c.empty()
    return np.array([r['tt'], r['te'], r['ee']]) * TCMB**2, t


def five(f, k1, k2, e):      # 5-point first derivative from values at offsets k1*e and k2*e = 2 k1*e
    return (f[-k2] - 8 * f[-k1] + 8 * f[k1] - f[k2]) / (12 * e)


if __name__ == '__main__':
    t0 = time.time()
    th0 = theta(0.0, FID['h'])
    # constraint slope, 5-point at fixed h / fixed omega_cdm
    e1, e2 = 1e-3, 1e-3
    t_om = (theta(-2 * e1, FID['h']) - 8 * theta(-e1, FID['h']) + 8 * theta(e1, FID['h']) - theta(2 * e1, FID['h'])) / (12 * e1)
    t_h = (theta(0, FID['h'] - 2 * e2) - 8 * theta(0, FID['h'] - e2) + 8 * theta(0, FID['h'] + e2) - theta(0, FID['h'] + 2 * e2)) / (12 * e2)
    slope = -t_om / t_h
    # h on the fixed-theta_s curve at every stencil point
    hk = {0: FID['h']}
    for k in OFFS:
        hk[k] = brentq(lambda h: theta(k * EPS, h) - th0, 0.6, 0.75, xtol=1e-12, rtol=1e-15)
    print('100theta_s = %.10f, dh/domega_m|theta = %.6f (dlnh/dlnomega_m = %.5f)' %
          (th0, slope, slope * (FID['omega_b'] + FID['omega_cdm'] + 0.00064420) / FID['h']))
    print('  check: (h(+e)-h(-e))/2e = %.6f' % ((hk[1] - hk[-1]) / (2 * EPS)), ' %.0fs' % (time.time() - t0))

    # CLASS spectra
    jobs = [(k, 'ref') for k in OFFS] + [(k, 'low') for k in [0, -2, -1, 1, 2]]   # C_l at the fiducial (norm only): LOW
    with Pool(2) as p:
        res = p.map(spectra, chunksize=1, iterable= [(k * EPS, hk[k], REF if w == 'ref' else LOW) for k, w in jobs])
    C = {j: r for j, r in zip(jobs, res)}
    thmax = max(abs(r[1] - th0) for r in res)
    print('CLASS runs done %.0fs; max |100theta_s - target| = %.1e' % (time.time() - t0, thmax))
    ref = {k: C[(k, 'ref')][0] for k in OFFS}
    low = {k: C[(k, 'low')][0] for k in [0, -2, -1, 1, 2]}
    d_cls = five(ref, 1, 2, EPS)
    d_cls_step = five(ref, 2, 4, 2 * EPS)
    d_cls_low = five(low, 1, 2, EPS)

    # emulator
    emu = spt_emu.SPT(EMU_DIR)
    il = slice(0, L - 1)                                      # emulator ell starts at 2 -> ell 2..2508
    assert emu.ell[0] == 2 and emu.ell[L - 2] == L
    pv = lambda dom, h: np.array([FID['omega_b'], FID['omega_cdm'] + dom, h, FID['tau_reio'], FID['n_s'], FID['ln10A']])
    pad = lambda a: np.concatenate([np.zeros((3, 2)), a], axis=1)       # index = ell, like CLASS
    E = {k: pad(np.array([emu.predict(pv(k * EPS, hk[k]))[s][il] for s in ('tt', 'te', 'ee')])) for k in [0] + OFFS}
    d_num = five(E, 1, 2, EPS)
    d_num_step = five(E, 2, 4, 2 * EPS)
    J = emu.jacobian(pv(0, FID['h']))
    d_an = pad(np.array([J[s][1][1][il] + slope * J[s][1][2][il] for s in ('tt', 'te', 'ee')]))

    np.savez(os.path.join(HERE, 'dcl_domegam_spt_vs_class.npz'), ell=np.arange(L + 1),
             C_class=low[0], C_spt=E[0], d_an=d_an, d_num=d_num, d_num_step=d_num_step,
             d_cls=d_cls, d_cls_step=d_cls_step, d_cls_low=d_cls_low,
             slope=slope, theta_s=th0, eps=EPS, h_points=np.array([[k, hk[k]] for k in [0] + OFFS]),
             fid=str(FID), ref_prec=str(REF), low_prec=str(LOW))
    print('saved, %.0fs total' % (time.time() - t0))
