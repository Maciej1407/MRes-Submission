#!/bin/bash -l


#$ -l h_rt=48:0:0
#$ -l h_vmem=32G
# tip: pass -t on qsub; see examples below
#$ -N KDCONTROL
#$ -cwd
#$ -t 1-360

# --- activate env ---
source /etc/profile.d/modules.sh
module load python/3.9
source mvenv_py39/bin/activate
python3 -c 'import sys, numpy as np; print("Python:", sys.version.split()[0], "NumPy:", np.__version__)'

# Positional args
GRID_SIZE="$1"; N_CELLS="$2"; ENV="$3"
MIG=0; DMETHOD=auto; SIM_TIME=300000
ROOT="HH2_S${GRID_SIZE}C${N_CELLS}_${ENV}"


# -------- pairs (range-matched within [5e-05, 1e-03]--------
consumption_rate_pairs=(
  "5e-05,5e-05"
  "1e-04,1e-04"
  "1.5e-04,1.5e-04"
  "2e-04,2e-04"
  "3e-04,3e-04"
  "4.5e-04,4.5e-04"
  "6e-04,6e-04"
  "1e-03,1e-03"
  "5e-05,1e-04"
  "5e-05,1.5e-04"
  "5e-05,2e-04"
  "5e-05,3e-04"
  "5e-05,4.5e-04"
  "5e-05,6e-04"
  "5e-05,1e-03"
  "1e-04,2e-04"
  "1e-04,3e-04"
  "1e-04,4.5e-04"
  "1e-04,6e-04"
  "1e-04,1e-03"
  "1.5e-04,3e-04"
  "1.5e-04,4.5e-04"
  "1.5e-04,6e-04"
  "1.5e-04,1e-03"
  "2e-04,4.5e-04"
  "2e-04,6e-04"
  "2e-04,8e-04"
  "3e-04,6e-04"
  "3e-04,8e-04"
  "4.5e-04,6e-04"
)

diffusion_rates=(0.1 0.25 0.5 0.75)

epochs=(0 1 2)

run_task() {
  IFS=',' read -r fR sR <<< "$1"
  diff="$2"
  ep="$3"
  python3 -u Simulation_macrophages_hpc.py \
    -env "$ENV" -fR "$fR" -sR "$sR" -diff "$diff" -ep "$ep" \
    -Nx "$GRID_SIZE" -Ny "$GRID_SIZE" -cells "$N_CELLS" \
    -mig "$MIG" -dmethod "$DMETHOD" -root "$ROOT" -Simulation_time "$SIM_TIME"
}

if [[ -n "${SGE_TASK_ID:-}" ]]; then
  R_COUNT=${#consumption_rate_pairs[@]}
  D_COUNT=${#diffusion_rates[@]}
  E_COUNT=${#epochs[@]}
  block=$(( D_COUNT * E_COUNT ))
  idx=$(( SGE_TASK_ID - 1 ))
  r=$(( idx / block ))
  rem=$(( idx % block ))
  d=$(( rem / E_COUNT ))
  e=$(( rem % E_COUNT ))
  run_task "${consumption_rate_pairs[$r]}" "${diffusion_rates[$d]}" "${epochs[$e]}"
else
  for r_pair in "${consumption_rate_pairs[@]}"; do
    for diff in "${diffusion_rates[@]}"; do
      for ep in "${epochs[@]}"; do
        run_task "$r_pair" "$diff" "$ep"
      done
    done
  done
fi
