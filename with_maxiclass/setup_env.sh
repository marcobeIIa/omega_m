#!/bin/bash
# Create the 'maxiclass' conda env (clone of 'class') and build mAxiCLASS in it.
# Run once per machine that makes mocks (Hive login node is fine, ~2 min):
#   bash ~/omega_m/with_maxiclass/setup_env.sh          (Mac: ~/uni/omega_m/...)
# Then:  conda activate maxiclass && cd ~/omega_m && bash with_maxiclass/make_mocks.sh
#
# - clones env 'class' (numpy etc.), adds cython, removes the pip classy 3.4.1 from the
#   CLONE only (the wrapper loads mAxiCLASS by path; the 'class' env is untouched)
# - clones mAxiCLASS (branch mAxiCLASShoot) into with_maxiclass/maxiclass if missing
# - builds it in place with this env's python (never pip-installs it)
set -e
ENV=maxiclass
SRC_ENV=${SRC_ENV:-class}
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../with_maxiclass

# ---------- conda ----------
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
    echo "env $ENV exists, reusing it"
else
    conda create -y -n "$ENV" --clone "$SRC_ENV"
fi
conda activate "$ENV"
pip uninstall -y classy >/dev/null 2>&1 || true
pip install "cython>=3" setuptools

# ---------- mAxiCLASS ----------
cd "$HERE"
[[ -d maxiclass ]] || git clone -b mAxiCLASShoot https://github.com/marcobeIIa/mAxiCLASS.git maxiclass
cd maxiclass
make clean >/dev/null 2>&1 || true
make -j4 class libclass.a
cd python
rm -rf build classy.c classy.cpp
CC=gcc python setup.py build

# ---------- check ----------
cd "$HERE"
python - <<'EOF'
import sys
sys.path.insert(0, "class/python")
import classy
print("mAxiCLASS python module:", classy.REAL_CLASSY_FILE)
EOF
echo "done: conda activate $ENV, then bash with_maxiclass/make_mocks.sh from omega_m/"
