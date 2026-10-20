#!/bin/bash
# Launch 8 independent MontePython chains as a Slurm job array.
#
# Submit from anywhere on Hive (make the log folder once first: mkdir -p ~/omega_m/logs):
#   sbatch ~/omega_m/slurm/run_chains.sbatch
#   PARAM=two_bins.param MODEL=fid_two_bins_null.dat sbatch ~/omega_m/slurm/run_chains.sbatch
#
# Quick test (1 chain, 200 steps, 20 min):
#   NSTEPS=200 RUN_TAG=test sbatch --array=1 -t 0:20:00 ~/omega_m/slurm/run_chains.sbatch
#
#SBATCH --job-name=mp_chains
#SBATCH --account=leknoxgrp
#SBATCH --partition=high
#SBATCH --array=1-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/mcbella/omega_m/logs/%x_%A_%a.out

# ===================== EDIT THESE (or override on the sbatch line) =====================
PIPELINE=${PIPELINE:-with_class}     # with_cosmopower | with_class
PARAM=${PARAM:-one_bin.param}            # file in ~/omega_m/$PIPELINE/
MODEL=${MODEL:-lcdm.dat}     # file in ~/omega_m/$MODEL_FROM/model/
MODEL_FROM=${MODEL_FROM:-$PIPELINE}       # pipeline that made the mock data (default: same as PIPELINE)
COVMAT=${COVMAT:-fid.covmat}              # file in ~/omega_m/models/  (set COVMAT=none to skip)
NSTEPS=${NSTEPS:-100000}
RUN_TAG=${RUN_TAG:-}                      # optional suffix, e.g. RUN_TAG=v2 for a fresh output folder
# =======================================================================================

ROOT=$HOME/omega_m
CHAIN=${SLURM_ARRAY_TASK_ID:-1}

case "$PIPELINE" in
    with_cosmopower) ENV=cosmopower ;;
    with_cp_lcdm) ENV=cp_lcdm ;;
    with_class)      ENV=class ;;
    with_class_mnu)  ENV=class ;;
    *) echo "ERROR: PIPELINE must be with_cosmopower, with_cp_lcdm, with_class or with_class_mnu (got '$PIPELINE')"; exit 1 ;;
esac

case "$MODEL_FROM" in
    with_cosmopower|with_class|with_cp_lcdm|with_class_mnu|with_maxiclass) ;;   # with_maxiclass: mocks only
    *) echo "ERROR: MODEL_FROM must be with_cosmopower, with_cp_lcdm, with_class, with_class_mnu or with_maxiclass (got '$MODEL_FROM')"; exit 1 ;;
esac

PARAM_PATH=$ROOT/$PIPELINE/$PARAM
MODEL_PATH=$ROOT/$MODEL_FROM/model/$MODEL
CONF_PATH=$ROOT/$PIPELINE/default.conf
# Cross runs (mock from the other pipeline) get "_from_<code>" in the folder name.
[[ "$MODEL_FROM" != "$PIPELINE" ]] && FROM_TAG=_from_${MODEL_FROM#with_} || FROM_TAG=
OUT=$ROOT/$PIPELINE/chains/model_${MODEL%.dat}${FROM_TAG}__param_${PARAM%.param}${RUN_TAG:+__$RUN_TAG}

# ---------- safety checks ----------
for f in "$PARAM_PATH" "$CONF_PATH"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done
# MontePython silently WRITES a new fiducial if the file is missing -> refuse instead.
[[ -f "$MODEL_PATH" ]] || { echo "ERROR: fiducial $MODEL_PATH not found (make it with models/make_fiducial.py first)"; exit 1; }
if [[ "$COVMAT" != none ]]; then
    COV_PATH=$ROOT/models/$COVMAT
    [[ -f "$COV_PATH" ]] || { echo "ERROR: covmat $COV_PATH not found"; exit 1; }
    COV_ARG=(-c "$COV_PATH")
else
    COV_ARG=()
fi

# ---------- environment ----------
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV" || { echo "ERROR: could not activate conda env $ENV"; exit 1; }

# Use only the cores Slurm gave us (no oversubscribing shared nodes)
NCPU=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NCPU            # CLASS
export TF_NUM_INTRAOP_THREADS=$NCPU     # TensorFlow / CosmoPower
export TF_NUM_INTEROP_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH=$ROOT/$PIPELINE/class/python   # this pipeline's classy only

# ---------- param file with the chosen fiducial ----------
# Every '<likelihood>.fiducial_file = ...' line in the .param (active or commented out) is
# dropped and one line pointing at MODEL is appended. The likelihood name is read from
# data.experiments (first entry).
LIKELIHOOD=$(grep -E "^[[:space:]]*data\.experiments[[:space:]]*=" "$PARAM_PATH" \
             | head -1 | sed -E "s/.*\[[[:space:]]*['\"]([^'\"]+)['\"].*/\1/")
[[ -n "$LIKELIHOOD" ]] || { echo "ERROR: could not read data.experiments from $PARAM_PATH"; exit 1; }

TMP_PARAM=$(mktemp --suffix=.param)
trap 'rm -f "$TMP_PARAM"' EXIT
sed -E "/^[[:space:]]*#?[[:space:]]*[A-Za-z0-9_]+\.fiducial_file[[:space:]]*=/d" "$PARAM_PATH" > "$TMP_PARAM"
printf "\n# Fiducial set by run_chains.sbatch\n%s.fiducial_file = '%s'\n" "$LIKELIHOOD" "$MODEL_PATH" >> "$TMP_PARAM"

# ---------- avoid 8 chains creating the same folder at once ----------
mkdir -p "$OUT"
if [[ -f "$OUT/log.param" ]]; then
    # Existing folder: MontePython will reuse its log.param, so make sure it matches.
    grep -q "$MODEL_PATH" "$OUT/log.param" \
        || { echo "ERROR: $OUT already exists with a different fiducial. Use a new RUN_TAG."; exit 1; }
elif [[ "$CHAIN" != 1 ]]; then
    echo "Chain $CHAIN waiting for chain 1 to write log.param ..."
    for _ in $(seq 360); do [[ -f "$OUT/log.param" ]] && break; sleep 5; done
    [[ -f "$OUT/log.param" ]] || { echo "ERROR: log.param never appeared (did chain 1 fail?)"; exit 1; }
    sleep 10
fi

# ---------- run ----------
echo "Host $(hostname) | env $ENV | $NCPU cpus | chain $CHAIN | $NSTEPS steps"
echo "param  $PARAM_PATH"
echo "model  $MODEL_PATH"
echo "output $OUT"

cd "$ROOT/montepython_fake"
python montepython/MontePython.py run \
    -p "$TMP_PARAM" -o "$OUT" -N "$NSTEPS" \
    --conf "$CONF_PATH" --chain-number "$CHAIN" "${COV_ARG[@]}"
