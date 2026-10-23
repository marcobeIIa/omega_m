#!/bin/bash
# Redo of the EDE runs (2026-10-22): the earlier ones were fitted to a wrong-h mock.
# mAxiCLASS 2-axion EDE mock (with_maxiclass/model/ede2ax_pdh0_mean.dat) fitted with the
# lcdm v1 emulator pipeline, for every omega_m model. Same output names as before:
#   ~/omega_m/with_cp_lcdm/chains/model_ede2ax_pdh0_mean_from_maxiclass__param_<param>/
#
# Run on Hive, after deleting the old folders:  bash ~/omega_m/slurm/submit_ede_redo.sh

SCRIPT=$HOME/omega_m/slurm/run_chains.sbatch
ROOT=$HOME/omega_m
MOCK=ede2ax_pdh0_mean.dat
CHAINS=${CHAINS:-1-8}
TIME=${TIME:-12:00:00}                       # emulator chains: ~3 h for 100000 steps
PARAMS=${PARAMS:-"one_bin three_bins poly five_bins ten_bins"}

mkdir -p "$ROOT/logs"

# Refuse to start if the mock is still the wrong-h version (TT at l=2: ~944 wrong, ~1013 right).
TT2=$(awk '!/^#/ && $1==2 {print $2; exit}' "$ROOT/with_maxiclass/model/$MOCK")
if ! awk -v x="$TT2" 'BEGIN{exit !(x > 1000)}'; then
    echo "STOP: $MOCK has TT(l=2) = $TT2 (expected ~1013): still the wrong-h mock."; exit 1
fi

for P in $PARAMS; do
    OUT=$ROOT/with_cp_lcdm/chains/model_${MOCK%.dat}_from_maxiclass__param_$P
    if [[ -e "$OUT" ]]; then
        echo "SKIP $P: $OUT still exists (delete the old run first)"; continue
    fi
    [[ "$P" == poly ]] && COV=poly_fisher.covmat || COV=fid.covmat
    PIPELINE=with_cp_lcdm MODEL_FROM=with_maxiclass MODEL=$MOCK PARAM=$P.param COVMAT=$COV \
        sbatch --job-name=ede_$P --array=$CHAINS -t $TIME "$SCRIPT"
done

squeue -u "$USER"
