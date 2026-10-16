# Debug check #1b — is the high-ℓ CLASS/CosmoPower gap a settings issue?

Point: omega_b 0.0224, omega_cdm 0.12, h 0.674, tau 0.054, n_s 0.965, ln10^10A_s 3.044. Lensed spectra everywhere.
Emulator = our classy stand-in, 1 bin (CosmoPower CP_paper CMB networks).

## 1. What the emulators were trained on
- CosmoPower paper (arXiv:2106.03846, Sec. 4.1): training set of ~5·10^5 spectra made with **CAMB**, not CLASS
  (the tutorial files are also named `camb_tt_training_*.npz`). CAMB version and accuracy settings are not given.
- Only omega_b, omega_cdm, h, tau, n_s, ln10^10A_s are inputs; everything else is fixed at unstated values.
- Accuracy metric (Eq. 11–13 and tutorial code): |ΔC_ℓ| / σ_CMB with σ_CMB = sqrt(2/(f_sky(2ℓ+1))) (C_ℓ + N_ℓ^SO goal ILC), f_sky = 0.4.
  Claim: 99% of test spectra < 0.1 σ_CMB for parameters within 5σ of Planck; up to ~0.2 σ_CMB over the full training box (App. C).

## 2. What CLASS assumes when you don't say (checked from explanatory.ini and the run itself)
- Massless neutrinos: N_ur = 3.044, N_ncdm = 0 (derived: Neff = 3.044, m_ncdm_tot = 0).
- YHe from BBN (0.2453 here); HyRec recombination; tanh reionization (width 0.5, He reionization at z = 3.5); T_cmb 2.7255 K.
- non_linear = none, so the lensing potential is LINEAR. Lensing applied only because we set lensing = yes.
- delta_l_max set automatically (l_max_scalars/3), so asking for l_max_scalars = 3000 gives lensed spectra accurate up to 2508.
- CAMB's python defaults differ: one massive neutrino of 0.06 eV, nnu ≈ 3.03 total, Recfast, linear lensing unless lens_potential_accuracy ≥ 1.

## 3. CLASS's own numerical precision is not the issue
Default CLASS vs the high-precision preset `cl_ref.pre` (8.7 min run):

| | ℓ=30 | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|
| TT default/ref − 1 | +0.02% | −0.08% | −0.05% | −0.03% | −0.05% | −0.04% |
| EE default/ref − 1 | +0.07% | +0.14% | −0.02% | −0.01% | −0.04% | −0.08% |

Also: l_max_scalars 2508→5000, k_max_tau0_over_l_max 1.0–1.8 and accurate_lensing = 1 move high-ℓ TT by ≤ 0.1%.

## 4. Emulator vs different reference codes/settings — relative difference, emulator/ref − 1 [%]

| reference | spec | ℓ=30 | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|
| CLASS 3.4, massless ν, linear lensing | TT | +0.03 | −0.02 | −0.22 | −0.34 | −0.71 | −0.78 |
| | EE | +0.46 | +0.23 | −0.02 | −0.61 | −0.28 | −0.95 |
| CLASS 3.4, mν = 0.06 | TT | +0.01 | −0.26 | +0.15 | +0.53 | −0.70 | +0.41 |
| CLASS 3.4, halofit lensing | TT | +0.03 | −0.01 | −0.22 | −0.27 | −0.91 | −1.77 |
| CLASS 2.9.4, massless ν, linear | TT | +0.03 | +0.00 | −0.20 | −0.31 | −0.55 | −0.52 |
| CAMB 2.0.4, mν = 0, linear lensing | TT | +0.05 | −0.10 | −0.19 | −0.31 | −0.64 | −0.56 |
| | EE | +0.47 | +0.36 | +0.00 | −0.55 | −0.27 | −0.86 |
| CAMB 2.0.4, mν = 0.06, linear lensing | TT | +0.02 | −0.36 | +0.17 | +0.58 | −0.66 | +0.65 |
| CAMB 2.0.4, mν = 0, NL lensing (lpa = 1) | TT | +0.04 | −0.08 | −0.19 | −0.23 | −0.82 | −1.49 |

Massive neutrinos produce oscillating residuals (a peak-position shift), and nonlinear lensing makes the tail worse.
The emulator is consistent with massless ν and linear lensing. Under those settings CLASS and CAMB agree with each other to ~0.1–0.2%,
and both sit 0.6–0.8% above the emulator at ℓ ≳ 2000.

## 5. Same gap in the paper's own metric (|ΔC|/σ_CMB, f_sky = 0.4, SO goal ILC noise)

| reference | spec | ℓ 40–799 med / max | 800–1599 med / max | 1600–2508 med / max |
|---|---|---|---|---|
| CLASS 3.4 default | TT | 0.004 / 0.038 | 0.064 / 0.108 | 0.168 / 0.230 |
| | EE | 0.021 / 0.067 | 0.043 / 0.140 | 0.097 / 0.178 |
| CAMB, mν = 0, linear lensing | TT | 0.008 / 0.038 | 0.059 / 0.100 | 0.138 / 0.186 |
| | EE | 0.019 / 0.074 | 0.039 / 0.133 | 0.089 / 0.168 |

At the Planck best fit (the centre of their "5σ Planck" box) TT reaches 0.15–0.19 σ_CMB above ℓ≈1600. That is about 2× the
paper's "99% < 0.1 σ_CMB" headline, and matches the ~0.2 σ_CMB they quote over the full training box (App. C).
The public CP_paper networks are trained on that full box, so this level of error is plausibly what they deliver.
The remaining unknowns are the exact CAMB version and accuracy settings used for training.
