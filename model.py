"""
model.py

Binned-omega_m LCDM spectrum, built on top of a pretrained CosmoPower
Cl emulator.

WHAT THIS DOES
--------------
You give it:
  - a base cosmology dict (all the usual LCDM params EXCEPT omega_m,
    e.g. omega_b, h, tau_reio, n_s, ln10^{10}A_s)
  - an array `l_switch` of length n_bins: the *upper* ell edge of each
    bin. Bin 0 runs from ell.min() up to l_switch[0]; bin 1 runs from
    l_switch[0] to l_switch[1]; ...; bin (n_bins-1) runs from
    l_switch[n_bins-2] up to l_switch[-1] (== ell.max(), typically).
  - an array `omega_m_bins` of length n_bins: the omega_m value to use
    in each of those bins.

It calls the CosmoPower emulator n_bins times (once per bin, each time
with that bin's omega_m and the shared base cosmology), and for each
call keeps only the Cl(ell) values that fall inside that bin's ell
range. The output is the concatenation of these pieces -> a single
Cl(ell) array covering the full ell range, but internally "patched"
together from n_bins different cosmologies.

NO smoothing or continuity is enforced across bin edges -- there will
generically be a jump in Cl at each l_switch value. This is
intentional, per spec, for now.

IMPORTANT -- parameter names in the actual pretrained network:
The public CosmoPower TT network (trained_models/CP_paper/CMB/cmb_TT_NN)
does NOT take omega_m as an input. Its parameters are:
    omega_b, omega_cdm, h, tau_reio, n_s, ln10^{10}A_s
(run `cp_nn.parameters` after loading to confirm for whatever network
you use). So this wrapper does the one conversion that's actually
needed -- and it's simple addition, not the Omega_m*h^2 business from
before, because omega_m and omega_cdm here are both physical
("little omega") densities:

    omega_cdm = omega_m - omega_b

omega_b is one of your *shared* base_params (fixed across all bins),
so for each bin we just subtract it off the bin's omega_m to get the
omega_cdm value to feed the network. You keep calling `.predict(...)`
with real omega_m values; the omega_cdm translation happens
internally, right before each emulator call.
"""

import numpy as np


class BinnedOmegaMEmulator:
    def __init__(self, cp_emulator, base_params, cdm_param_name="omega_cdm"):
        """
        Parameters
        ----------
        cp_emulator : cosmopower_NN or cosmopower_PCAplusNN
            A *loaded* (restore=True) CosmoPower network. Must expose:
              - `.modes` : the native ell grid the network predicts on
              - `.predictions_np(params_dict)` : dict of param arrays
                 -> array of shape (n_samples, n_ell)
        base_params : dict
            All shared LCDM parameters, using whatever key names your
            specific network was trained with. MUST include "omega_b"
            (used to convert omega_m -> omega_cdm for each bin), e.g.
            {"omega_b": 0.0224, "h": 0.674, "tau_reio": 0.054,
             "n_s": 0.965, "ln10^{10}A_s": 3.044}
            Do NOT include cdm_param_name in here -- it gets set
            per-bin from omega_m_bins.
        cdm_param_name : str
            The key name the emulator expects for the CDM density
            input. Defaults to "omega_cdm" (matches the public CMB TT
            network). Check `cp_nn.parameters` if unsure.
        """
        self.cp = cp_emulator
        self.ell = np.asarray(cp_emulator.modes)
        self.base_params = dict(base_params)
        self.cdm_param_name = cdm_param_name

        if "omega_b" not in self.base_params:
            raise ValueError(
                "base_params must include 'omega_b' -- it's needed to "
                "convert each bin's omega_m into omega_cdm for the emulator."
            )

    def _bin_masks(self, l_switch):
        """
        Build boolean masks over self.ell for n_bins = len(l_switch)
        bins, with upper edges given by l_switch and an implicit lower
        edge of self.ell.min() for the first bin.
        """
        l_switch = np.asarray(l_switch)
        n_bins = len(l_switch)

        edges = np.concatenate(([self.ell.min()], l_switch))
        # edges has length n_bins + 1: edges[i] -> edges[i+1] is bin i

        masks = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                m = (self.ell >= lo) & (self.ell <= hi)
            else:
                m = (self.ell > lo) & (self.ell <= hi)
            masks.append(m)

        n_assigned = sum(m.sum() for m in masks)
        if n_assigned != len(self.ell):
            raise ValueError(
                f"l_switch does not partition the full ell range: "
                f"{n_assigned} of {len(self.ell)} ell values assigned. "
                f"Make sure l_switch is increasing and l_switch[-1] >= ell.max()."
            )
        return masks

    def predict(self, l_switch, omega_m_bins):
        """
        Parameters
        ----------
        l_switch : array_like, length n_bins
            Upper ell edge of each bin (see class docstring).
        omega_m_bins : array_like, length n_bins
            omega_m value to use in each bin, same ordering as l_switch.

        Returns
        -------
        ell : array, the native ell grid (unchanged)
        cl_patched : array, patched Cl(ell) in LINEAR space (already
            un-logged), same length as ell
        """
        l_switch = np.asarray(l_switch)
        omega_m_bins = np.asarray(omega_m_bins)
        if len(l_switch) != len(omega_m_bins):
            raise ValueError(
                f"l_switch (len {len(l_switch)}) and omega_m_bins "
                f"(len {len(omega_m_bins)}) must be the same length"
            )

        masks = self._bin_masks(l_switch)
        cl_patched = np.zeros_like(self.ell, dtype=float)

        omega_b = self.base_params["omega_b"]

        for mask, om in zip(masks, omega_m_bins):
            params = dict(self.base_params)
            params[self.cdm_param_name] = om - omega_b  # omega_m -> omega_cdm

            # cosmopower expects each parameter as a length-1 (or batched) array
            cp_params = {k: np.atleast_1d(v) for k, v in params.items()}
            # predictions_np returns log10(Cl) for this network (it was
            # trained on log-spectra) -- exponentiate to get linear Cl
            # before patching. Equivalent to cp.ten_to_predictions_np(),
            # done manually here since we only need one row at a time.
            log_cl_full = self.cp.predictions_np(cp_params)[0]  # shape (n_ell,)
            cl_full = 10.0 ** log_cl_full

            cl_patched[mask] = cl_full[mask]

        return self.ell, cl_patched
