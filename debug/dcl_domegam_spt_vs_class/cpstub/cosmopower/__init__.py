"""Minimal numpy-only stand-in for cosmopower 0.2.0's cosmopower_NN / cosmopower_PCAplusNN
(restore from .pkl, predictions_np, dict_to_ordered_arr_np), so the pipeline's classy.py runs
without TensorFlow. Same attribute names, same stored dtypes (float32) and the same numpy
expressions as cosmopower's forward_pass_np / predictions_np."""
import pickle
import numpy as np


class _LW(list):
    pass


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('tensorflow') and name == 'ListWrapper':
            return _LW
        if mod == 'numpy.core.multiarray':
            mod = 'numpy._core.multiarray'
        return super().find_class(mod, name)


def _load(fn):
    with open(fn + '.pkl', 'rb') as f:
        return _U(f).load()


class _Base:
    def dict_to_ordered_arr_np(self, d):
        return np.stack([d[k] for k in self.parameters], axis=1)

    def forward_pass_np(self, x):
        layers = [(x - self.parameters_mean_) / self.parameters_std_]
        for i in range(self.n_layers - 1):
            act = np.dot(layers[-1], self.W_[i]) + self.b_[i]
            layers.append((self.betas_[i] + (1. - self.betas_[i]) * 1. / (1. + np.exp(-self.alphas_[i] * act))) * act)
        return np.dot(layers[-1], self.W_[-1]) + self.b_[-1]


class cosmopower_NN(_Base):
    def __init__(self, restore=True, restore_filename=None, **kw):
        (W, b, a, be, self.parameters_mean_, self.parameters_std_, self.features_mean_, self.features_std_,
         self.n_parameters, self.parameters, self.n_modes, self.modes, self.n_hidden, self.n_layers,
         self.architecture) = _load(restore_filename)
        self.W_, self.b_, self.alphas_, self.betas_ = [list(map(np.asarray, z)) for z in (W, b, a, be)]
        self.parameters = list(self.parameters)

    def predictions_np(self, d):
        return self.forward_pass_np(self.dict_to_ordered_arr_np(d)) * self.features_std_ + self.features_mean_


class cosmopower_PCAplusNN(_Base):
    def __init__(self, restore=True, restore_filename=None, **kw):
        (W, b, a, be, self.parameters_mean_, self.parameters_std_, self.pca_mean_, self.pca_std_,
         self.features_mean_, self.features_std_, self.parameters, self.n_parameters, self.modes, self.n_modes,
         self.n_pcas, self.pca_transform_matrix_, self.n_hidden, self.n_layers, self.architecture) = _load(restore_filename)
        self.W_, self.b_, self.alphas_, self.betas_ = [list(map(np.asarray, z)) for z in (W, b, a, be)]
        self.parameters = list(self.parameters)

    def predictions_np(self, d):
        return np.dot(self.forward_pass_np(self.dict_to_ordered_arr_np(d)) * self.pca_std_ + self.pca_mean_,
                      self.pca_transform_matrix_) * self.features_std_ + self.features_mean_
