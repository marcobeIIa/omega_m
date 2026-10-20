#!/bin/bash
# Smooth delta omega_m(l) model (with_cp_lcdm/poly.param, l_c 400) on two mocks.
# Run on Hive:  bash ~/omega_m/slurm/submit_poly_tests.sh
#
#   1) NULL TEST: the usual CLASS mock with matching physics (1 nu 0.06 eV + HMcode),
#      with_class_mnu/model/lcdm.dat (omega_m 0.15). Expect dwm_b, dwm_c consistent with 0.
#      -> ~/omega_m/with_cp_lcdm/chains/model_lcdm_from_class_mnu__param_poly/
#   2) mAxiCLASS mock, with_maxiclass/model/$MAXI_MOCK
#      -> ~/omega_m/with_cp_lcdm/chains/model_<mock>_from_maxiclass__param_poly/
#      plus the plain-LCDM baseline (one_bin.param) on the same mock
#      -> ~/omega_m/with_cp_lcdm/chains/model_<mock>_from_maxiclass__param_one_bin/
#
# Switches (env vars):  RUN_NULL=0 / RUN_MAXI=0 skip a block;  BASELINE=0 skips the
# one_bin runs;  MAXI_MOCK=other.dat picks another mAxiCLASS mock;  CHAINS=1-2 fewer chains.
#   e.g. RUN_NULL=0 MAXI_MOCK=ede2ax_pdh0_mean.dat bash ~/omega_m/slurm/submit_poly_tests.sh

SCRIPT=$HOME/omega_m/slurm/run_chains.sbatch
ROOT=$HOME/omega_m
CHAINS=${CHAINS:-1-8}
TIME=${TIME:-12:00:00}               # emulator chains take ~3 h for 100000 steps
RUN_NULL=${RUN_NULL:-1}
RUN_MAXI=${RUN_MAXI:-1}
BASELINE=${BASELINE:-1}
MAXI_MOCK=${MAXI_MOCK:-maxiclass.dat}

mkdir -p "$ROOT/logs"

# Refuse at submit time if a mock is missing (run_chains.sbatch would also refuse,
# but only after the job has waited in the queue).
need() { [[ -f "$1" ]] || { echo "SKIP: $1 not found"; return 1; }; }

# ---------------- 1) null test: CLASS (mnu + HMcode) mock ----------------
if [[ "$RUN_NULL" == 1 ]] && need "$ROOT/with_class_mnu/model/lcdm.dat"; then
    PIPELINE=with_cp_lcdm MODEL_FROM=with_class_mnu MODEL=lcdm.dat PARAM=poly.param COVMAT=poly_fisher.covmat \
        sbatch --job-name=poly_on_classmock --array=$CHAINS -t $TIME "$SCRIPT"
    # Its LCDM baseline is model_lcdm_from_class_mnu__param_one_bin (submit_cross.sh job 1);
    # uncomment if that run doesn't exist yet:
    #PIPELINE=with_cp_lcdm MODEL_FROM=with_class_mnu MODEL=lcdm.dat PARAM=one_bin.param COVMAT=fid.covmat \
    #    sbatch --job-name=lcdm_on_classmock --array=$CHAINS -t $TIME "$SCRIPT"
fi

# ---------------- 2) mAxiCLASS mock ----------------
if [[ "$RUN_MAXI" == 1 ]] && need "$ROOT/with_maxiclass/model/$MAXI_MOCK"; then
    TAG=${MAXI_MOCK%.dat}
    PIPELINE=with_cp_lcdm MODEL_FROM=with_maxiclass MODEL=$MAXI_MOCK PARAM=poly.param COVMAT=poly_fisher.covmat \
        sbatch --job-name=poly_on_$TAG --array=$CHAINS -t $TIME "$SCRIPT"
    if [[ "$BASELINE" == 1 ]]; then
        PIPELINE=with_cp_lcdm MODEL_FROM=with_maxiclass MODEL=$MAXI_MOCK PARAM=one_bin.param COVMAT=fid.covmat \
            sbatch --job-name=lcdm_on_$TAG --array=$CHAINS -t $TIME "$SCRIPT"
    fi
fi

squeue -u "$USER"
