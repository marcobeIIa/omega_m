# omega_m binning -- status slides (October 2026)

- `omega_m_binning.pdf` -- the deck (16 slides + 5 backup). Rebuild: `pdflatex omega_m_binning.tex` (twice), figures in `figs/`.
- `figs/` -- every figure as PDF.
- `data/` -- numbers quoted on the slides: `summary.json` (GetDist means/68/95%, chi2_min, acceptance, R-1, LCDM test for the 9 CosmoPower runs + the CLASS run),
  `optimal_bins.json` (Fisher optimum, models/optimal_bins.py output), `emubias.json` (Fisher forecast: CosmoPower model fitted to a CLASS mock),
  `edges.json` (C_l jumps at bin edges, real CLASS), `theta_pulls.json`, `bench_mcmc.json`.
- `scripts/` -- how the figures/numbers were made (run in the Claude sandbox; paths point at copies of this folder under /mnt/user-data/uploads/omega_m --
  change `U` in runs.py / fisherlib.py to the omega_m root to rerun). Chains: with_cosmopower/chains/model_{lcdm,2bins,3bins}__param_{one_bin,two_bins,three_bins},
  with_class/chains/model_lcdm__param_one_bin; burn-in 0.3, same conventions as analyse_chain.ipynb. Fisher: models/fisher_derivatives.npz + models/optimal_bins.py.
  CLASS = class_public v3.4.0; CosmoPower = with_cosmopower/class/python/classy.py.
