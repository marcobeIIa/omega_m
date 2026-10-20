#!/bin/bash
# Fishing: mAxiCLASS 2-axion EDE mock (with_maxiclass/model/ede2ax_pdh0_mean.dat,
# Bella+ 2026 PDH0 means) fitted with the lcdm v1 emulator pipeline.
# Run on Hive:  bash ~/omega_m/slurm/submit_ede.sh
#
#   1) poly.param (LCDM + smooth delta omega_m(l), l_c 400)
#      -> ~/omega_m/with_cp_lcdm/chains/model_ede2ax_pdh0_mean_from_maxiclass__param_poly/
#   2) one_bin.param (plain LCDM, the baseline the poly run is compared with)
#      -> ~/omega_m/with_cp_lcdm/chains/model_ede2ax_pdh0_mean_from_maxiclass__param_one_bin/
#
# Before: make the mock (bash ~/omega_m/with_maxiclass/make_mocks.sh) or sync it from the Mac.

SCRIPT=$HOME/omega_m/slurm/run_chains.sbatch
CHAINS=1-8
MOCK=ede2ax_pdh0_mean.dat

mkdir -p "$HOME/omega_m/logs"

# 1) poly
PIPELINE=with_cp_lcdm MODEL_FROM=with_maxiclass MODEL=$MOCK PARAM=poly.param COVMAT=poly_fisher.covmat \
    sbatch --job-name=poly_on_ede2ax --array=$CHAINS -t 12:00:00 "$SCRIPT"

# 2) LCDM baseline (comment out if not wanted)
PIPELINE=with_cp_lcdm MODEL_FROM=with_maxiclass MODEL=$MOCK PARAM=one_bin.param COVMAT=fid.covmat \
    sbatch --job-name=lcdm_on_ede2ax --array=$CHAINS -t 12:00:00 "$SCRIPT"

squeue -u "$USER"
