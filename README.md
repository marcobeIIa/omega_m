# MontePython + CosmoPower (binned omega_m, fixed binning) -- step 2

## What's in this folder

- `classy.py` -- stand-in for the real CLASS Python wrapper. MontePython
  does `from classy import Class`; if this file's directory is earlier
  on `sys.path` (or `PYTHONPATH`) than any real classy install, that
  import resolves here instead, and MontePython drives CosmoPower
  without knowing the difference. No MontePython source changes needed.
- `binned_omegam.param` -- MontePython input parameter file wiring the
  shared LCDM params + `Om_bin0..N` to what `classy.py` expects.

## Before running

1. **Fill in the nuisance-parameter block.** I left this as a
   placeholder in `binned_omegam.param` -- open whichever
   `base_2018_plikHM_*.param` ships with your MontePython install
   (under `montepython/input/`) and copy its nuisance-parameter lines
   verbatim into `binned_omegam.param`. Those are internal to the
   Planck likelihood's foreground/calibration model and unrelated to
   the omega_m binning.

2. **Set `data.experiments`** in `binned_omegam.param` to whatever your
   install actually calls its Planck 2018 likelihood
   (`ls montepython/likelihoods/` to check the exact folder name).

3. **Set environment variables** for the fixed (non-sampled)
   configuration `classy.py` reads:
   ```bash
   export COSMOPOWER_TT_PATH=/path/to/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN
   export L_SWITCH=800,1600,2508
   ```
   `L_SWITCH` must have as many entries as you have `Om_bin*` parameters
   in the `.param` file (3, to match the example above).

4. **Put the shim ahead of any real classy on PYTHONPATH:**
   ```bash
   export PYTHONPATH=/path/to/montepython_shim:$PYTHONPATH
   ```
   If you don't have real classy/CLASS installed in this environment at
   all, this step is moot but doesn't hurt.

## Run

```bash
python montepython/MontePython.py run \
    -p /path/to/montepython_shim/binned_omegam.param \
    -o /path/to/some/output/dir \
    -N 10000
```

Start with a small `-N` (e.g. 100) first, purely to confirm the shim
doesn't crash and the chain file has sane-looking numbers, before
committing to a full run.

## Things that are very likely to break on first try, in rough order

1. **`data.experiments` name wrong** -- fix to match your install
   (see step 2 above).
2. **Missing nuisance parameters** -- if you didn't copy the full
   nuisance block, the Planck likelihood will complain about missing
   parameters it expects.
3. **TE/EE not implemented** -- `classy.py` currently returns zero
   arrays for TE/EE/BB/PP. If your chosen Planck likelihood uses
   TE/EE (Planck 2018's does), you need to load the corresponding
   CosmoPower TE/EE networks and patch them the same way TT is
   patched in `classy.py`'s `compute()`. Flag if you want this built
   out now, or want to first confirm the TT-only skeleton runs at all.
4. **Derived parameter errors** -- if the likelihood or a prior asks
   for a derived parameter other than `H0`/`Omega_m`,
   `get_current_derived_parameters()` will raise `NotImplementedError`
   loudly (by design, rather than returning something meaningless).

## Reminder on what this model can and can't sensibly answer

Because omega_m varies by ell bin, there's no single background
cosmology -- `Omega_m` as reported here is a placeholder tied to bin 0,
not a real cosmological quantity. Treat any run using this shim as
testing the CMB-spectrum-level statistics only (steps 2-3 of your
plan), not as a physically consistent full-cosmology inference.
