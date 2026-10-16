#!/bin/bash
# Cross-check: mock data from one code, sampled with the other code.
# Run on Hive:  bash ~/omega_m/slurm/submit_cross.sh
#
#   1) CLASS mock (with_class/model/lcdm.dat)       sampled with CosmoPower, one_bin.param
#      -> ~/omega_m/with_cosmopower/chains/model_lcdm_from_class__param_one_bin/
#   2) CosmoPower mock (with_cosmopower/model/lcdm.dat) sampled with CLASS, one_bin.param
#      -> ~/omega_m/with_class/chains/model_lcdm_from_cosmopower__param_one_bin/

SCRIPT=$HOME/omega_m/slurm/run_chains.sbatch
CHAINS=1-8        # chains per run (e.g. 1-2 for just two chains)

mkdir -p "$HOME/omega_m/logs"

# 1) CLASS mock, CosmoPower sampler
PIPELINE=with_cp_lcdm MODEL_FROM=with_class MODEL=lcdm.dat PARAM=one_bin.param \
    sbatch --job-name=cp_on_classmock --array=$CHAINS "$SCRIPT"

# 2) CosmoPower mock, CLASS sampler
#PIPELINE=with_class MODEL_FROM=with_cosmopower MODEL=lcdm.dat PARAM=one_bin.param \
#    sbatch --job-name=class_on_cpmock --array=$CHAINS -c 4 --mem=8G "$SCRIPT"

squeue -u "$USER"
