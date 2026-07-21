# fake_planck_highl_TTTEEE

A toy MontePython likelihood, built on top of your `classy` cosmopower
shim, for testing whether a pipeline can recover an injected step in
Omega_m at l = 1250. It is **not** the real Planck likelihood -- it's a
minimal Gaussian diagonal-covariance stand-in.

## What's in here

```
binning.py                          shared binning helper (imported by generate_fake_data.py)
generate_fake_data.py               builds the mock "observed" bandpowers
montepython_likelihood/
  fake_planck_highl_TTTEEE/
    __init__.py                     the MontePython Likelihood class
    fake_planck_highl_TTTEEE.data   its config file
```

## 1. Generate the fake data

Run this in the same environment you use for `test_classy_shim.py`
(i.e. `classy` must resolve to your shim, not real CLASS):

```bash
export COSMOPOWER_TT_PATH=/path/to/cmb_TT_NN
export L_SWITCH=1250          # single switch point -> Om_bin0 (l<1250), Om_bin1 (l>=1250)
python generate_fake_data.py --outdir fake_planck_data --seed 42
```

This calls the shim with standard LCDM (omega_b=0.0224, h=0.674,
tau_reio=0.054, n_s=0.965, ln10^10 A_s=3.044) but `Om_bin0 = 0.310` for
l < 1250 and `Om_bin1 = 0.322` for l >= 1250 -- a ~4% step. Edit
`OMEGA_M_LOW_L` / `OMEGA_M_HIGH_L` at the top of `generate_fake_data.py` to
change the size of the injected signal (set them equal to inject *no*
signal, as a null test).

It bins the theory into l=30..2508 bandpowers of width 30, adds toy
Gaussian noise (cosmic-variance-like + a rising instrumental floor above
l~1600 -- see `toy_sigma_dl()`), and writes:

```
fake_planck_data/fake_planck_highl_TTTEEE_bandpowers.dat
```

with columns `l_center Dl_TT sigma_TT Dl_TE sigma_TE Dl_EE sigma_EE`
(uK^2). Use `--noiseless` to write the exact model instead of a noisy
realization, if you want a clean recovery test first.

**Note:** the noise model is a rough approximation for pipeline-testing
purposes, not a real Planck covariance -- there's no TT-TE-EE
cross-covariance and no beam/foreground modeling. If your shim only
returns `tt` from `raw_cl()`, the script will fill TE/EE with NaN and
tell you to set `use_te = False`, `use_ee = False` in the `.data` file.

## 2. Install the likelihood

```bash
cp -r montepython_likelihood/fake_planck_highl_TTTEEE  <montepython>/montepython/likelihoods/
cp fake_planck_data/fake_planck_highl_TTTEEE_bandpowers.dat  <montepython>/data/
```

(Or point `fake_planck_highl_TTTEEE.data_directory` in the `.data` file at
an absolute path instead of moving the file into `data/`.)

Then add `fake_planck_highl_TTTEEE` to your run's likelihood list, e.g.:

```bash
python montepython/MontePython.py run \
    -p your_run.param -o chains/fake_planck_test \
    --likelihood fake_planck_highl_TTTEEE
```

with `Om_bin0`, `Om_bin1` (and whatever else your shim wants, matching the
`L_SWITCH` env var used to generate the data) as varied/fixed parameters
in `your_run.param`, alongside the usual `omega_b`, `h`, `tau_reio`,
`n_s`, `ln10^{10}A_s`.

## 3. Options in `fake_planck_highl_TTTEEE.data`

- `use_tt` / `use_te` / `use_ee` -- toggle which spectra enter chi2.
- `lmin` / `lmax` -- optional extra ell cuts on top of the data file's range.
- `vary_calibration` -- if `True`, adds an `A_planck` nuisance parameter
  (model scaled by `A_planck**2`, Gaussian prior `A_planck_prior_center`
  ± sqrt(`A_planck_prior_variance`)) the way the real Planck likelihoods
  do. Leave `False` for a plain chi2 with no nuisance parameters. If you
  turn it on, remember to also add `A_planck` to your `.param` file.

## Recovering the injected step

Because the fake data has genuinely different Omega_m below vs above
l=1250, a fit that only allows a single, scale-independent Omega_m should
show up as a poor fit / biased best-fit Omega_m, while a fit using the
same `Om_bin0`/`Om_bin1` (+ matching `L_SWITCH`) parameterization as the
injection should recover `OMEGA_M_LOW_L` and `OMEGA_M_HIGH_L` from
`generate_fake_data.py`. That contrast is the point of building it this
way rather than with a single flat Omega_m.
