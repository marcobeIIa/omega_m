"""Independent check that every CLASS stencil point sits on the fixed-theta_s curve: recompute CLASS 100*theta_s
at (omega_cdm + k eps, h_k) (thermodynamics-only run; theta_s does not depend on C_l precision settings)."""
import numpy as np
import compute_derivs as cd
z = np.load('dcl_domegam_spt_vs_class.npz')
rows = []
for k, h in z['h_points']:
    k = int(k); t = cd.theta(k * cd.EPS, h)
    rows.append([k, cd.FID['omega_b'] + cd.FID['omega_cdm'] + k * cd.EPS + 0.0006441359, h, t, t - float(z['theta_s'])])
rows = np.array(sorted(rows))
np.save('theta_check.npy', rows)
for r in rows: print('%+d  omega_m %.6f  h %.8f  100theta_s %.10f  diff %+.1e' % tuple(r))
