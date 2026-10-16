#!/bin/bash
# Submit the full set of runs. Run on Hive from anywhere:  bash ~/omega_m/slurm/submit_all.sh
#
#   CosmoPower: 3 fiducials x 3 param files = 9 jobs of 8 chains (1 CPU each)
#   CLASS:      lcdm.dat x one_bin.param    = 1 job  of 8 chains (4 CPUs each)
#
# Output: ~/omega_m/<pipeline>/chains/model_<model>__param_<param>/

SCRIPT=$HOME/omega_m/slurm/run_chains.sbatch
THROTTLE=4        # max CosmoPower chains running at once per job (8 = all at once)

mkdir -p "$HOME/omega_m/logs"

# ---------------- CosmoPower: 3 x 3 ----------------
for M in lcdm 2bins 3bins; do
    for P in one_bin two_bins three_bins; do
        PIPELINE=with_cosmopower MODEL=$M.dat PARAM=$P.param \
            sbatch --job-name=cp_${M}_${P} --array=1-8%$THROTTLE "$SCRIPT"
    done
done

# ---------------- CLASS: lcdm data, one-bin fit ----------------
PIPELINE=with_class MODEL=lcdm.dat PARAM=one_bin.param \
    sbatch --job-name=class_lcdm_one_bin -c 4 --mem=8G "$SCRIPT"

squeue -u "$USER"
