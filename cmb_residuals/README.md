# cmb_residuals/

CMB residuals (TT, TE, EE, dd) of fitted models with respect to a mock.

- `cmb_residuals.ipynb` — set `MOCK` (mock .dat) and `MODELS` ({label: MP output folder}, paths relative to `omega_m/`),
  `MODE = "bestfit"` (min −lnL sample) or `"posterior"` (`N_SAMPLES` weighted draws after `BURN_IN` → median + 68/95% bands),
  `RESIDUAL` (`"sigma"`, `"frac"`, `"dl"`), `BIN_WIDTH` (int, 1/None, or an array of edges), `SPECTRA`, `XSCALE`.
  Cell 2 computes the spectra, cell 3 plots (re-run cell 3 alone to change residual kind / binning).
  PNGs go to `cmb_residuals/plots/`.
- `residuals.py` — helpers: reads `log.param` (params × scale, fixed `l_switch`/`l_c`, `cosmo_arguments`, likelihood settings),
  picks the newest chain root in the folder (override with `ROOTS`), loads the run's own `classy.py`
  (`data.path['cosmo']` remapped to this machine) under a unique module name, reads the mock + noise file as MontePython does.

Notes
- Spectra are recomputed with the pipeline each run was sampled with, so the kernel needs that pipeline's dependencies
  (`with_cp_lcdm`: the `cp_lcdm` env). Tested with `with_cp_lcdm` runs only.
- Check (sandbox, 2026-10-23): the MontePython χ² recomputed from these spectra at the best fit equals 2·min(−lnL) of the
  chains to 1e-5 (EDE mock: one_bin 5.4254, poly 5.0342, ten_bins 7.9061).
- `sigma` residuals use the diagonal Gaussian error of each spectrum (cosmic variance + noise, f_sky), not the full
  TT–TE–EE covariance of the likelihood.
