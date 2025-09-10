#!/bin/bash -l

#$ -l h_rt=3:0:0
#$ -l mem=8G
#$ -N FinalRuns
#$ -cwd
#$ -t 1-1800

set -euo pipefail

# --- activate env ---
source /etc/profile.d/modules.sh
module load python/3.9
source mvenv_py39/bin/activate
python3 -c 'import sys, numpy as np; print("Python:", sys.version.split()[0], "NumPy:", np.__version__)'

# Positional args
GRID_SIZE="$1"; N_CELLS="$2"; ENV="$3"
MIG=0; DMETHOD=auto; SIM_TIME=300000
ROOT="Final_S${GRID_SIZE}C${N_CELLS}_${ENV}"

# -------- pairs (same as before) --------

consumption_rate_pairs=(
  "5e-05,5e-05" "5e-05,7.5e-05" "5e-05,1e-04" "5e-05,2e-04" "5e-05,3e-04" "5e-05,5e-04" "5e-05,1e-03" "5e-05,2e-03"
  "7.5e-05,7.5e-05" "7.5e-05,1e-04" "7.5e-05,2e-04" "7.5e-05,3e-04" "7.5e-05,5e-04" "7.5e-05,1e-03" "7.5e-05,2e-03"
  "1e-04,1e-04" "1e-04,2e-04" "1e-04,3e-04" "1e-04,5e-04" "1e-04,1e-03" "1e-04,2e-03"
  "2e-04,2e-04" "2e-04,3e-04" "2e-04,5e-04" "2e-04,1e-03" "2e-04,2e-03"
  "3e-04,3e-04" "3e-04,5e-04" "3e-04,1e-03" "3e-04,2e-03"
  "5e-04,5e-04" "5e-04,1e-03" "5e-04,2e-03"
  "1e-03,1e-03" "1e-03,2e-03"
  "2e-03,2e-03"

  "0.0001,0.01" "0.001,0.01" "0.0001,1" "0.1,0.5" "0.00015,0.00015" "0.00025,0.0005"
  "0.0005,0.0015" "0.00025,0.002" "0.00025,0.00025" "0.0004,0.0004" "0.0001,0.0006" "0.0006,0.0006"
  "0.00025,0.001" "0.0005,0.00075" "0.0015,0.0015" "0.0025,0.0025" "0.0001,0.0004"
  "5e-05,5e-03" "1e-04,5e-03" "2e-04,5e-03" "5e-04,5e-03" "1e-03,5e-03" "2e-03,5e-03" "5e-03,5e-03"
)


# ---- choose which half of diffusion to run, for space limit purposes ----
DIFF_PICK="${DIFF_PICK:-all}"   # first | second | all
_full=(0.1 0.25 0.5 0.75 1 1.2)
case "${DIFF_PICK,,}" in
  first|a)   diffusion_rates=("${_full[@]:0:3}") ;;   # 0.1,0.25,0.5
  second|b)  diffusion_rates=("${_full[@]:3:3}") ;;   # 0.75,1,1.2
  all|*)     diffusion_rates=("${_full[@]}") ;;
esac


# --- verification ---
declare -A _seen; _dups=0
for cr in "${consumption_rate_pairs[@]}"; do
  [[ -n "${_seen[$cr]:-}" ]] && { echo "DUPLICATE: $cr" >&2; ((_dups++)); } || _seen[$cr]=1
done
(( _dups == 0 )) || { echo "ERROR: duplicates found"; exit 10; }


epochs=(0 1 2 3 4)

[[ "${#epochs[@]}" -eq 5 && " ${epochs[*]} " =~ " 0 " && " ${epochs[*]} " =~ " 1 " && " ${epochs[*]} " =~ " 2 " && " ${epochs[*]} " =~ " 3 " && " ${epochs[*]} " =~ " 4 " ]] || { echo "ERROR: epochs must be {0,1,2,3,4}"; exit 11; }


# diffusion verification
csv="$(printf "%s," "${diffusion_rates[@]}")"
[[ "$csv" == "0.1,0.25,0.5," || "$csv" == "0.75,1,1.2," || "$csv" == "0.1,0.25,0.5,0.75,1,1.2," ]] || { echo "ERROR: diffusion set invalid: {${csv%,}}"; exit 12; }

R_COUNT=${#consumption_rate_pairs[@]}   # 88
D_COUNT=${#diffusion_rates[@]}          # 3 or 6
E_COUNT=${#epochs[@]}                   # 3
TOTAL=$(( R_COUNT * D_COUNT * E_COUNT )) # 792 or 1584

echo "Plan (DIFF_PICK=${DIFF_PICK}): R=${R_COUNT} × D=${D_COUNT} × E=${E_COUNT} = TOTAL ${TOTAL}"
echo "Submit array with: -t 1-${TOTAL}"

block=$(( D_COUNT * E_COUNT ))

run_task() {
  local r="$1" d="$2" e="$3"
  IFS=',' read -r fR sR <<< "${consumption_rate_pairs[$r]}"
  local diff="${diffusion_rates[$d]}" ep="${epochs[$e]}"
  echo "[START $(date)] r=$((r+1))/${R_COUNT} d=$((d+1))/${D_COUNT} e=$((e+1))/${E_COUNT} :: fR=${fR} sR=${sR} D=${diff} ep=${ep}"
  if python3 -u Simulation_macrophages_hpc.py \
      -env "${ENV}" -fR "${fR}" -sR "${sR}" -diff "${diff}" -ep "${ep}" \
      -Nx "${GRID_SIZE}" -Ny "${GRID_SIZE}" -cells "${N_CELLS}" \
      -mig "${MIG}" -dmethod "${DMETHOD}" -root "${ROOT}" -Simulation_time "${SIM_TIME}"
  then
      echo "[OK   $(date)] fR=${fR} sR=${sR} D=${diff} ep=${ep}"
  else
      rc=$?; echo "[FAIL $(date)] rc=${rc} :: fR=${fR} sR=${sR} D=${diff} ep=${ep}" >&2; exit $rc
  fi
}

if [[ -n "${SGE_TASK_ID:-}" ]]; then
  idx=$(( SGE_TASK_ID - 1 ))
  (( idx >= 0 && idx < TOTAL )) || { echo "ERROR: index ${SGE_TASK_ID} not in 1..${TOTAL}"; exit 14; }
  r=$(( idx / block )); rem=$(( idx % block )); d=$(( rem / E_COUNT )); e=$(( rem % E_COUNT ))
  run_task "$r" "$d" "$e"
else
  for ((r=0; r<R_COUNT; r++)); do
    for ((d=0; d<D_COUNT; d++)); do
      for ((e=0; e<E_COUNT; e++)); do
        run_task "$r" "$d" "$e"
      done
    done
  done
fi

