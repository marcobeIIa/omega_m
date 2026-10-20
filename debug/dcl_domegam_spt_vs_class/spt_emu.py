"""SPT_high_accuracy LCDM emulators (Balkenhol+2022, CosmoPower format) in plain numpy, float64.

predict(p)       -> dict tt/te/ee (emulator output, l = 2..5000)
jacobian(p)      -> (values, d values / d params)   exact (analytic) derivative of the network,
                    forward-mode chain rule through every layer: no finite differences.
Parameter order: ombh2, omch2, h, tau, ns, logA.
The .pkl files reference tensorflow's ListWrapper only; a list stand-in is used so TF is not needed.
"""
import os, pickle
import numpy as np

PARAMS = ['ombh2', 'omch2', 'h', 'tau', 'ns', 'logA']


class _LW(list):
    pass


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('tensorflow') and name == 'ListWrapper':
            return _LW
        if mod == 'numpy.core.multiarray':
            mod = 'numpy._core.multiarray'
        return super().find_class(mod, name)


def _load(path):
    with open(path, 'rb') as f:
        return _U(f).load()


class Net:
    def __init__(self, path):
        o = _load(path)
        self.pca = 'PCA' in os.path.basename(path)
        f64 = lambda L: [np.asarray(x, dtype=np.float64) for x in L]
        self.W, self.b, self.alpha, self.beta = f64(o[0]), f64(o[1]), f64(o[2]), f64(o[3])
        self.pmean, self.pstd = np.float64(o[4]), np.float64(o[5])
        if self.pca:
            self.pca_mean, self.pca_std = np.float64(o[6]), np.float64(o[7])
            self.fmean, self.fstd = np.float64(o[8]), np.float64(o[9])
            assert list(o[10]) == PARAMS
            self.ell = np.asarray(o[12]); self.T = np.float64(o[15])
        else:
            self.fmean, self.fstd = np.float64(o[6]), np.float64(o[7])
            assert list(o[9]) == PARAMS
            self.ell = np.asarray(o[11])
        self.log = not self.pca            # TT/EE NN emulate log10; TE PCAplusNN is linear

    def _forward(self, p, jac):
        x = (np.asarray(p, float) - self.pmean) / self.pstd
        t = np.diag(1.0 / self.pstd) if jac else None          # d x / d p   (6 x 6)
        n = len(self.W)
        for i in range(n):
            z = x @ self.W[i] + self.b[i]
            if jac:
                tz = t @ self.W[i]
            if i < n - 1:
                a, be = self.alpha[i], self.beta[i]
                s = 1.0 / (1.0 + np.exp(-a * z))
                x = (be + s * (1 - be)) * z
                if jac:
                    fp = be + (1 - be) * (s + a * z * s * (1 - s))   # d act / d z
                    t = tz * fp
            else:
                x = z
                if jac:
                    t = tz
        if self.pca:
            x = x * self.pca_std + self.pca_mean
            y = (x @ self.T) * self.fstd + self.fmean
            dy = ((t * self.pca_std) @ self.T) * self.fstd if jac else None
        else:
            y = x * self.fstd + self.fmean
            dy = t * self.fstd if jac else None
        if self.log:
            v = 10.0 ** y
            dv = np.log(10.0) * v * dy if jac else None
        else:
            v, dv = y, dy
        return v, dv

    def predict(self, p):
        return self._forward(p, False)[0]

    def jacobian(self, p):
        return self._forward(p, True)       # values (n_l,), derivs (6, n_l)


class SPT:
    def __init__(self, d):
        self.nets = {k: Net(os.path.join(d, f)) for k, f in
                     [('tt', 'cmb_lcdm_spt_TT_NN.pkl'), ('te', 'cmb_lcdm_spt_TE_PCAplusNN.pkl'),
                      ('ee', 'cmb_lcdm_spt_EE_NN.pkl')]}
        self.ell = self.nets['tt'].ell

    def predict(self, p):
        return {k: n.predict(p) for k, n in self.nets.items()}

    def jacobian(self, p):
        return {k: n.jacobian(p) for k, n in self.nets.items()}
