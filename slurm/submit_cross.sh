#!/bin/bash
# Cross-check: mock data from one code, sampled with the other code.
# Run on Hive:  bash ~/omega_m/slurm/submit_cross.sh
#
#   1) CLASS mock with 1 massive nu + HMcode (with_class_mnu/model/lcdm.dat)
#      sampled with the lcdm v1 emulators (with_cp_lcdm), one_bin.param
#      -> ~/omega_m/with_cp_lcdm/chains/model_lcdm_from_class_mnu__param_one_bin/
#   2) CosmoPower mock (with_cosmopower/model/lcdm.dat) sampled with CLASS, one_bin.param
#      -> ~/omega_m/with_class/chains/model_lcdm_from_cosmopower__param_one_bin/

SCRIPT=$HOME/omega_m/slurm/run_chains.sbatch
CHAINS=1-8        # chains per run (e.g. 1-2 for just two chains)

mkdir -p "$HOME/omega_m/logs"

# 1) CLASS (massive nu) mock, lcdm v1 emulator sampler (~3 h per chain)
#PIPELINE=with_cp_lcdm MODEL_FROM=with_class_mnu MODEL=lcdm.dat PARAM=one_bin.param \
#   sbatch --job-name=cp_on_classmock --array=$CHAINS -t 12:00:00 "$SCRIPT"

# 2) CosmoPower mock, CLASS sampler
PIPELINE=with_class_mnu MODEL_FROM=with_cp_lcdm MODEL=lcdm.dat PARAM=one_bin.param \
    sbatch --job-name=class_on_cpmock --array=$CHAINS -c 4 --mem=8G "$SCRIPT"

squeue -u "$USER"
