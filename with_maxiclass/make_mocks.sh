#!/bin/bash
# Mocks made with mAxiCLASS. Run from omega_m/ in an env with numpy (Hive: conda activate class):
#   bash with_maxiclass/make_mocks.sh
# Each takes ~10 s. Existing files are kept (add --force to overwrite).
set -e
cd "$(dirname "$0")/.."

# 2-axion EDE (n = 3), Bella+ 2026 (arXiv:2604.13535), PDH0 posterior means
# (Tables S1, S2): 100 omega_b 2.267, omega_cdm 0.133, H0 72.645, tau 0.060,
# ln10^10 A_s 3.067, n_s 0.988; f_ax 0.085 / 0.073 at log10 a_c -3.440 / -4.000;
# theta_i 2.8. omega_m = omega_b + omega_cdm + omega_nu = 0.02267 + 0.13300 + 0.00064 = 0.15631.
python models/make_fiducial.py --pipeline with_maxiclass \
    --omega_m 0.15631 --l_switch 2508 \
    --omega_b 0.02267 --h 0.72645 --tau_reio 0.060 --n_s 0.988 --lnAs 3.067 \
    --set N_mscf=2 n_axion_mscf=3,3 theta_ini_mscf=2.8,2.8 \
          fraction_maxion_ac=0.085,0.073 log10_maxion_ac=-3.440,-4.000 \
    --name ede2ax_pdh0_mean.dat "$@"

# LCDM control at the SAME base parameters (no EDE): not a good fit to anything,
# only for plotting what the EDE does at fixed (omega_b, omega_m, h, ...).
python models/make_fiducial.py --pipeline with_maxiclass \
    --omega_m 0.15631 --l_switch 2508 \
    --omega_b 0.02267 --h 0.72645 --tau_reio 0.060 --n_s 0.988 --lnAs 3.067 \
    --name lcdm_at_ede2ax_pdh0_base.dat "$@"
