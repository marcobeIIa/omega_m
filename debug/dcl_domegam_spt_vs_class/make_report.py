"""Figures and LaTeX tables for report/dcl_domegam_report.tex.

Inputs: dcl_domegam_spt_vs_class.npz (compute_derivs.py), dcl_domegam_lcdm_v1.npz (compute_v1.py),
theta_check.npy (theta_check.py). Outputs in report/: fig_derivs.pdf, fig_errors.pdf, fig_an_vs_num.pdf,
tab_*.tex, numbers.tex.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'report'); os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID, BAND = '#1a1a19', '#6b6a63', '#e5e4de', '#d9d7cf'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10, 'legend.fontsize': 8.5,
                     'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.bbox': 'tight', 'mathtext.fontset': 'cm', 'font.family': 'serif'})

z = np.load(os.path.join(HERE, 'dcl_domegam_spt_vs_class.npz'))
v = np.load(os.path.join(HERE, 'dcl_domegam_lcdm_v1.npz'))
th = np.load(os.path.join(HERE, 'theta_check.npy'))
ell = z['ell']; s = slice(2, None); l = ell[s].astype(float)
C = z['C_class']
D = {'cls': z['d_cls'], 'spt_num': z['d_num'], 'spt_an': z['d_an'],
     'v1_num': v['v1_num'], 'v1_an': v['v1_an'], 'v1_an_x': v['v1_an_exact']}
floor = np.maximum(np.abs(z['d_cls_step'] - D['cls']), np.abs(z['d_cls_low'] - D['cls']))
norm = np.array([C[0], np.sqrt(C[0] * C[2]), C[2]])
Dl = ell * (ell + 1) / (2 * np.pi)
SP = ['TT', 'TE', 'EE']
NORMLAB = [r'$C_\ell^{TT}$', r'$\sqrt{C_\ell^{TT}C_\ell^{EE}}$', r'$C_\ell^{EE}$']
NRAW = [r'C_\ell^{TT}', r'\sqrt{C_\ell^{TT}C_\ell^{EE}}', r'C_\ell^{EE}']


def RELLAB(j):   # model error Delta C_l = Delta(dC/domega_m) * delta omega_m, over C_l, per unit delta omega_m
    return r'$\Delta C_\ell^{%s,\,\rm model}$' % SP[j] + '\n' + r'$/\ (%s\ \delta\omega_m)$' % NRAW[j]


LAB_SPT = r'SPT emulator, analytic'
LAB_V1 = r'lcdm v1 emulator (with\_cp\_lcdm), analytic'
LAB_V1p = 'lcdm v1 emulator (with_cp_lcdm), analytic'


def setup(ax):
    for a in ax.flat:
        a.set_xscale('log'); a.set_xlim(2, 2508); a.axhline(0, color=MUTED, lw=0.6)


# ---------- Fig 1: the derivatives
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0), gridspec_kw={'wspace': 0.45})
for j in range(3):
    a = ax[j]
    a.plot(l, (Dl * D['cls'][j])[s], color=INK, lw=2.2, label='CLASS, numerical (5-point)')
    a.plot(l, (Dl * D['spt_an'][j])[s], color=BLUE, lw=1.0, ls=(0, (4, 2)), label='SPT emulator, analytic')
    a.plot(l, (Dl * D['v1_an'][j])[s], color=ORANGE, lw=1.0, ls=(0, (1, 1.2)), label=LAB_V1p)
    a.set_title(SP[j]); a.set_xlabel(r'multipole $\ell$')
    a.set_ylabel(r'$\partial D_\ell^{%s}/\partial\omega_m|_{\theta_s}$  [$\mu$K$^2$]' % SP[j])
setup(ax[None, :])
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.98))
fig.savefig(os.path.join(OUT, 'fig_derivs.pdf')); plt.close(fig)

# ---------- Fig 2: errors of the analytic derivatives vs CLASS (absolute, relative)
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, gridspec_kw={'hspace': 0.12, 'wspace': 0.55})
for j in range(3):
    a = ax[0, j]
    a.fill_between(l, -(Dl * floor[j])[s], (Dl * floor[j])[s], color=BAND, lw=0, label='CLASS numerical floor')
    a.plot(l, (Dl * (D['spt_an'][j] - D['cls'][j]))[s], color=BLUE, lw=0.8, label='SPT emulator, analytic $-$ CLASS')
    a.plot(l, (Dl * (D['v1_an'][j] - D['cls'][j]))[s], color=ORANGE, lw=1.5,
           label='lcdm v1 emulator (with_cp_lcdm), analytic $-$ CLASS')
    a.set_title(SP[j])
    a.set_ylabel(r'$\Delta\,\partial D_\ell^{%s}/\partial\omega_m$  [$\mu$K$^2$]' % SP[j])
    a = ax[1, j]
    a.fill_between(l, -(floor[j] / norm[j])[s], (floor[j] / norm[j])[s], color=BAND, lw=0)
    a.plot(l, ((D['spt_an'][j] - D['cls'][j]) / norm[j])[s], color=BLUE, lw=0.8)
    a.plot(l, ((D['v1_an'][j] - D['cls'][j]) / norm[j])[s], color=ORANGE, lw=1.5)
    a.set_ylabel(RELLAB(j))
    a.set_xlabel(r'multipole $\ell$')
setup(ax)
YL = 0.13
for j in range(3):
    ax[1, j].set_ylim(-YL, YL)
    for k, col, nm in (('v1_an', ORANGE, 'lcdm v1'), ('spt_an', BLUE, 'SPT')):
        r = ((D[k][j] - D['cls'][j]) / norm[j])[s]
        off = np.abs(r) > YL
        if off.any():
            print('off scale: %s %s: %d of %d l, all l <= %d, max |rel| %.2f' % (SP[j], nm, off.sum(), off.size, l[off].max(), np.abs(r).max()))
h_, l_ = ax[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.95))
fig.savefig(os.path.join(OUT, 'fig_errors.pdf')); plt.close(fig)

# ---------- Fig 3: analytic vs numerical, same emulator (checks the analytic machinery and the slope)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0), gridspec_kw={'wspace': 0.55})
for j in range(3):
    a = ax[j]
    a.plot(l, ((D['v1_an'][j] - D['v1_num'][j]) / norm[j])[s], color=ORANGE, lw=1.5,
           label=r'lcdm v1: analytic (pipeline fitted $s$) $-$ numerical')
    a.plot(l, ((D['v1_an_x'][j] - D['v1_num'][j]) / norm[j])[s], color=INK, lw=0.9,
           label=r'lcdm v1: analytic (exact CLASS $s$) $-$ numerical')
    a.plot(l, ((D['spt_an'][j] - D['spt_num'][j]) / norm[j])[s], color=BLUE, lw=0.8, ls=(0, (4, 2)),
           label=r'SPT: analytic (exact CLASS $s$) $-$ numerical')
    a.set_title(SP[j]); a.set_xlabel(r'multipole $\ell$')
    a.set_ylabel(RELLAB(j))
setup(ax[None, :])
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.98))
fig.savefig(os.path.join(OUT, 'fig_an_vs_num.pdf')); plt.close(fig)


# ---------- tables
LS = [2, 10, 30, 100, 200, 500, 800, 1000, 1500, 2000, 2500]


def sci(x, p=3):
    m, e = ('%.*e' % (p, x)).split('e')
    return r'$%s\times10^{%d}$' % (m, int(e))


for j in range(3):
    t = [r'\begin{tabular}{r S[table-format=-1.3e1] S[table-format=-1.3e1] S[table-format=-1.3e1] S[table-format=-1.3e1] S[table-format=-1.3e1] S[table-format=-1.3e1]}',
         r'\toprule',
         r'{$\ell$} & {$D_\ell^{%s}$} & {CLASS} & {SPT num.} & {SPT anal.} & {lcdm v1 num.} & {lcdm v1 anal.} \\' % SP[j],
         r' & {[$\mu$K$^2$]} & \multicolumn{5}{c}{$\partial D_\ell^{%s}/\partial\omega_m|_{\theta_s}$ [$\mu$K$^2$]} \\' % SP[j],
         r'\midrule']
    for L in LS:
        vals = [Dl[L] * C[j, L]] + [Dl[L] * D[k][j, L] for k in ('cls', 'spt_num', 'spt_an', 'v1_num', 'v1_an')]
        t.append('%d & ' % L + ' & '.join('%.3e' % x for x in vals) + r' \\')
    t += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tab_values_%s.tex' % SP[j]), 'w').write('\n'.join(t) + '\n')

    t = [r'\begin{tabular}{r S[table-format=-2.2] S[table-format=-2.2] S[table-format=-2.2] S[table-format=-2.3] S[table-format=-2.3] S[table-format=1.2]}',
         r'\toprule',
         r'& \multicolumn{3}{c}{analytic $-$ CLASS} & \multicolumn{2}{c}{analytic $-$ numerical} & {CLASS} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}',
         r'{$\ell$} & {SPT} & {lcdm v1} & {lcdm v1, exact $s$} & {SPT} & {lcdm v1} & {floor} \\',
         r'\midrule']
    for L in LS:
        r = np.array([D['spt_an'][j, L] - D['cls'][j, L], D['v1_an'][j, L] - D['cls'][j, L],
                      D['v1_an_x'][j, L] - D['cls'][j, L], D['spt_an'][j, L] - D['spt_num'][j, L],
                      D['v1_an'][j, L] - D['v1_num'][j, L], floor[j, L]]) / norm[j, L] * 100
        t.append('%d & %.2f & %.2f & %.2f & %.3f & %.3f & %.2f \\\\' % ((L,) + tuple(r)))
    t += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tab_rel_%s.tex' % SP[j]), 'w').write('\n'.join(t) + '\n')

# theta_s check table
t = [r'\begin{tabular}{r S[table-format=1.6] S[table-format=1.8] S[table-format=1.10] S[table-format=-1.1e-1]}', r'\toprule',
     r'{$k$} & {$\omega_m$} & {$h$} & {$100\,\theta_s$} & {$100\,\theta_s - 100\,\theta_s^{\rm fid}$} \\', r'\midrule']
for k, om, h, tt, d in th:
    t.append(r'%+d & %.6f & %.8f & %.10f & %.1e \\' % (k, om, h, tt, d))
t += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(OUT, 'tab_theta.tex'), 'w').write('\n'.join(t) + '\n')

# sqrt(chi2) for delta omega_m = 1 sigma (cosmic variance only, f_sky 0.57, 30 <= l <= 2508)
fs, dw, w = 0.57, 0.0012, slice(30, 2509)
t = [r'\begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]}', r'\toprule',
     r' & {TT} & {TE} & {EE} \\', r'\midrule']
rows = [('template itself (CLASS)', lambda j: D['cls'][j]),
        ('SPT analytic $-$ CLASS', lambda j: D['spt_an'][j] - D['cls'][j]),
        ('lcdm v1 analytic (pipeline) $-$ CLASS', lambda j: D['v1_an'][j] - D['cls'][j]),
        ('lcdm v1 analytic, exact $s$ $-$ CLASS', lambda j: D['v1_an_x'][j] - D['cls'][j]),
        ('lcdm v1 analytic (pipeline) $-$ numerical', lambda j: D['v1_an'][j] - D['v1_num'][j]),
        ('CLASS numerical floor', lambda j: floor[j])]
for name, f in rows:
    vals = []
    for j in range(3):
        var = ((C[0] * C[2] + C[1]**2) if j == 1 else 2 * C[j]**2) / ((2 * ell + 1) * fs)
        vals.append(np.sqrt(np.sum((f(j) * dw)[w]**2 / var[w])))
    t.append(name + ' & ' + ' & '.join('%.3f' % x for x in vals) + r' \\')
t += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(OUT, 'tab_chi2.tex'), 'w').write('\n'.join(t) + '\n')

# numbers used in the text
spt = __import__('spt_emu')
num = {
    'thetafid': '%.10f' % float(z['theta_s']),
    'slope': '%.6f' % float(z['slope']),
    'sexact': '%.5f' % float(v['s_exact']), 'spipe': '%.5f' % float(v['s_pipe']),
    'thetamax': '%.0e' % np.max(np.abs(th[:, 4])),
    'eps': '%.1e' % float(z['eps']),
}
with open(os.path.join(OUT, 'numbers.tex'), 'w') as f:
    for k, x in num.items():
        f.write('\\newcommand{\\%s}{%s}\n' % (k, x))
print(num)
