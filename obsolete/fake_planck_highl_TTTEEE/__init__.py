"""
fake_planck_highl_TTTEEE

A toy MontePython likelihood that compares the classy-shim theory spectrum
(see test_classy_shim.py) against a mock "observed" bandpower file produced
by generate_fake_data.py -- a standard LCDM model with Omega_m stepping to
a slightly different value above vs below l=1250.

This is NOT the real Planck likelihood. It's a minimal Gaussian,
diagonal-covariance stand-in meant for testing a pipeline / recovering an
injected signal, nothing more. In particular it has no beam/foreground/
calibration nuisance modeling beyond the single optional A_planck
parameter below, and no TT-TE-EE cross-covariance.

Drop this folder into montepython/likelihoods/, put
fake_planck_highl_TTTEEE_bandpowers.dat (from generate_fake_data.py) into
your data.path['data'] directory, and add 'fake_planck_highl_TTTEEE' to
your .param / run's likelihood list like any other likelihood.
"""
import os
import numpy as np
from montepython.likelihood_class import Likelihood


# --- binning helpers -------------------------------------------------------
# Deliberately duplicated from binning.py (rather than imported) so this
# plugin folder is self-contained, as MontePython likelihood folders
# usually are. If you change the binning in generate_fake_data.py, mirror
# the change here and re-generate the data file.

def make_bin_edges(lmin, lmax, delta_l):
    edges = np.arange(lmin, lmax + delta_l, delta_l, dtype=float)
    if edges[-1] < lmax:
        edges = np.append(edges, float(lmax))
    return edges


def bin_spectrum(ell, dl, bin_edges):
    ell = np.asarray(ell)
    dl = np.asarray(dl)
    centers, binned = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (ell >= lo) & (ell < hi)
        if not np.any(mask):
            continue
        centers.append(0.5 * (lo + hi))
        binned.append(np.mean(dl[mask]))
    return np.array(centers), np.array(binned)


# Must match generate_fake_data.py's LMIN/LMAX/DELTA_L.
LMIN, LMAX, DELTA_L = 30, 2508, 30


class fake_planck_highl_TTTEEE(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        datafile = os.path.join(self.data_directory, self.data_file)
        table = np.loadtxt(datafile)
        self.l_center = table[:, 0]
        self.dl_tt, self.sigma_tt = table[:, 1], table[:, 2]
        self.dl_te, self.sigma_te = table[:, 3], table[:, 4]
        self.dl_ee, self.sigma_ee = table[:, 5], table[:, 6]

        # Apply optional ell cuts on top of the file's native range.
        lmin = self.lmin if self.lmin is not None else -np.inf
        lmax = self.lmax if self.lmax is not None else np.inf
        self.mask = (self.l_center >= lmin) & (self.l_center <= lmax)

        self.bin_edges = make_bin_edges(LMIN, LMAX, DELTA_L)

        if self.vary_calibration:
            self.use_nuisance = ['A_planck']
        else:
            self.use_nuisance = []

        # Sanity check: if a spectrum column is all-NaN (shim didn't
        # produce it when the fake data was generated) but the .data file
        # still asks to use it, turn it off with a warning rather than
        # crashing every single MCMC step.
        for spec, dl in (('tt', self.dl_tt), ('te', self.dl_te), ('ee', self.dl_ee)):
            flag = getattr(self, 'use_%s' % spec)
            if flag and np.all(np.isnan(dl)):
                print("fake_planck_highl_TTTEEE: 'use_%s' was True but the "
                      "data file has no %s spectrum -- disabling it." % (spec, spec.upper()))
                setattr(self, 'use_%s' % spec, False)

    def get_cl_dict(self, cosmo):
        cl = cosmo.raw_cl()
        ell = np.asarray(cl['ell'], dtype=float)
        out = {}
        for spec in ('tt', 'te', 'ee'):
            if spec in cl:
                out[spec] = np.asarray(cl[spec]) * ell * (ell + 1) / (2 * np.pi) * 1e12
        out['ell'] = ell
        return out

    def loglkl(self, cosmo, data):

        cl = self.get_cl_dict(cosmo)
        ell = cl['ell']

        chi2 = 0.0

        if self.use_tt:
            _, model = bin_spectrum(ell, cl['tt'], self.bin_edges)
            chi2 += self._chi2_term(model, self.dl_tt, self.sigma_tt, data)

        if self.use_te:
            _, model = bin_spectrum(ell, cl['te'], self.bin_edges)
            chi2 += self._chi2_term(model, self.dl_te, self.sigma_te, data)

        if self.use_ee:
            _, model = bin_spectrum(ell, cl['ee'], self.bin_edges)
            chi2 += self._chi2_term(model, self.dl_ee, self.sigma_ee, data)

        if self.vary_calibration:
            a_planck = data.mcmc_parameters['A_planck']['current'] * \
                data.mcmc_parameters['A_planck']['scale']
            chi2 += (a_planck - self.A_planck_prior_center) ** 2 / self.A_planck_prior_variance

        return -0.5 * chi2

    def _chi2_term(self, model, obs, sigma, data):
        m = self.mask & np.isfinite(model) & np.isfinite(obs) & np.isfinite(sigma)
        if not np.any(m):
            return 0.0
        resid = obs[m] - model[m]
        if self.vary_calibration:
            a_planck = data.mcmc_parameters['A_planck']['current'] * \
                data.mcmc_parameters['A_planck']['scale']
            resid = obs[m] - model[m] * a_planck ** 2
        return np.sum((resid / sigma[m]) ** 2)
