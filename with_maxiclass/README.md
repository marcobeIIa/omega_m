# with_maxiclass — beyond-LCDM mocks from mAxiCLASS

Mock-making pipeline only ("fishing" mocks). mAxiCLASS never goes into
MontePython: its mocks are fitted with the other pipelines through
`MODEL_FROM=with_maxiclass` in `slurm/run_chains.sbatch` (see `slurm/submit_ede.sh`).

```
with_maxiclass/
  maxiclass/                 git clone of mAxiCLASS, branch mAxiCLASShoot (compiled in place)
  class/python/classy.py     binned-omega_m wrapper (same model/interface as with_class_mnu)
  model/                     mocks, same format as every other pipeline's model/
  make_mocks.sh              the recipe for the mocks in model/
  ede2ax_pdh0_mean.ini       the same EDE cosmology as a plain mAxiCLASS .ini
```

## Setup (once per machine that makes mocks)

```bash
cd ~/omega_m/with_maxiclass            # Mac: ~/uni/omega_m/with_maxiclass
git clone -b mAxiCLASShoot https://github.com/marcobeIIa/mAxiCLASS.git maxiclass
cd maxiclass
make -j4 class libclass.a              # C code + static library
cd python && python setup.py build     # -> python/build/lib.*/classy*.so (NOT installed)
```

- Do **not** run `make classy` / `make all` / `pip install .`: that pip-installs a package
  called `classy` into the active env and would replace classy 3.4.1 in Hive's `class`
  env (used by with_class / with_class_mnu). The wrapper loads the built `.so` by path.
- Build needs numpy + Cython (`pip install cython` in the env you use; Hive: `conda activate class`).
- mAxiCLASS finds its data files (BBN, HyRec) through the absolute path compiled into it,
  so build it on the machine that runs it. If Mutagen syncs this folder, ignore the build
  products (`with_maxiclass/maxiclass/build`, `.../python/build`, `.../libclass.a`,
  `.../class`, `.../output`) or clone separately on each machine.
- Another location: `export MAXICLASS_DIR=/path/to/mAxiCLASS`.

## Making mocks

```bash
cd ~/omega_m && bash with_maxiclass/make_mocks.sh      # ~10 s per mock
```
Header line 1 is the usual one (read by analyse_chain.ipynb); line 2 lists the extra CLASS
keys (`--set`), line 3 the EDE fractions/positions actually reached by the shooting.

## Model conventions
- omega_m = omega_b + omega_cdm + omega_nu (one 0.06 eV neutrino, omega_nu = 0.00064420),
  as in with_class_mnu / with_cp_lcdm. EDE is not part of omega_m.
- N_ur 2.0328, N_ncdm 1, m_ncdm 0.06, non_linear hmcode (= lcdm v1 emulators).
- N_mscf > 0 adds: shooting on, theta' = 0, attractor_ic_scf no, tol_shooting 1e-3,
  background_Nloga 1e5. Repo example .ini files use tol 1, which stops the shooting after
  one step (f_EDE 0.138 for a 0.10 target); 1e-4 does not converge.
- compute() checks the shooting: peak f_EDE within 1% and log10 a_c within 0.01 of the
  targets for each field, else CosmoComputationError.

## Checks (sandbox, 2026-10-18; mAxiCLASS v3.3.0 base, commit b5a3af9)
- No EDE (N_mscf 0) at omega_m 0.15 vs with_class_mnu/model/lcdm.dat (CLASS 3.4.1):
  sqrt(chi2) of the difference 0.05 (TT+N max 0.014%).
- EDE mock reproducibility: shooting tolerance moves a_c by ~1e-3 -> sqrt(chi2) ~0.15
  between runs with slightly different inputs; background_Nloga default -> 1e5: 0.13,
  1e5 -> 3e5: 0.015.
- ede2ax_pdh0_mean.ini through ./class and the python wrapper agree to 5e-13.
