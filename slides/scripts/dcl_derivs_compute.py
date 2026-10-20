# d C_l / d omega_m at the benchmark point, and the same derivative with each other LCDM parameter shifted by +-3%.
# Real CLASS (classy 3.4.1 in the sandbox). Writes ../data/dcl_derivs.npz, read by fig_dcl_domegam.py. ~1 min.
import os
import classy, numpy as np, json, time
L = 2508; l = np.arange(L + 1); T = 2.7255e6
FID = dict(omega_b=0.0224, h=0.674, tau_reio=0.054, n_s=0.965, lnAs=3.044, omega_m=0.15)
DOM = 0.0015          # half step in omega_m for the derivative
def cls(p):
    c = classy.Class()
    c.set(dict(output='tCl pCl lCl', lensing='yes', l_max_scalars=L, omega_b=p['omega_b'], h=p['h'],
               tau_reio=p['tau_reio'], n_s=p['n_s'], A_s=np.exp(p['lnAs']) / 1e10, omega_cdm=p['omega_m'] - p['omega_b']))
    c.compute(); r = c.lensed_cl(L); c.struct_cleanup(); c.empty()
    return np.array([r['tt'] * T**2, r['te'] * T**2, r['ee'] * T**2, r['pp']])
def deriv(p):
    up = dict(p, omega_m=p['omega_m'] + DOM); dn = dict(p, omega_m=p['omega_m'] - DOM)
    return (cls(up) - cls(dn)) / (2 * DOM)
def shifted(name, f):
    p = dict(FID)
    if name == 'A_s':
        p['lnAs'] = FID['lnAs'] + np.log(f)                      # 3% in A_s
    else:
        p[name] = FID[name] * f                                   # omega_b at fixed omega_m (omega_cdm compensates)
    return p
t = time.time()
out = {'l': l, 'C0': cls(FID), 'D0': deriv(FID)}
for name in ['h', 'omega_b', 'A_s', 'n_s', 'tau_reio']:
    out['Dp_' + name] = deriv(shifted(name, 1.03)); out['Dm_' + name] = deriv(shifted(name, 0.97))
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dcl_derivs.npz'), **out); print('done %.0fs' % (time.time() - t))
