#!/bin/bash -l
#$ -o HH_FIN_Attmetrics_$JOB_ID.out
#$ -e HH_FIN_Attmetrics_$JOB_ID.err
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l mem=32G

# Usage:
#   qsub hpc_attractant_metrics.sh 500 50 Maze
# Args:
#   $1 = GRID_SIZE (e.g., 500)
#   $2 = N_CELLS   (e.g., 50)
#   $3 = ENV       (e.g., Maze)

set -euo pipefail

# -------- Read arguments --------
GRID_SIZE=${1:?need GRID_SIZE, e.g. 500}
N_CELLS=${2:?need N_CELLS, e.g. 50}
ENV=${3:?need ENV, e.g. Maze}
LABEL=$4

# Base path to your runs (edit if needed)
BASE="/home/dmcbmkm/Scratch/pra_run"
ROOT="Fin_S${GRID_SIZE}C${N_CELLS}_${ENV}/Maze"
RUN_ROOT="${BASE}/${ROOT}"

# Output CSV names (include identifiers for convenience)
OUT_SUMMARY="CS_AttractantMetrics_summary_S${GRID_SIZE}C${N_CELLS}_${ENV}_${LABEL}.csv"
OUT_SUBPOPS="CS_AttractantMetrics_subpops_S${GRID_SIZE}C${N_CELLS}_${ENV}_${LABEL}.csv"

# -------- Environment (as per your working setup) --------
source /etc/profile.d/modules.sh
module load python/3.9
# your working venv:
source mvenv_py39/bin/activate

# -------- Run metrics extractor --------
echo "[START $(date)] Host: $(hostname)"
echo "[INFO] ROOT=${RUN_ROOT}"
echo "[INFO] OUT_SUMMARY=${OUT_SUMMARY}"
echo "[INFO] OUT_SUBPOPS=${OUT_SUBPOPS}"

python3 compute_attractant_metrics.py "${RUN_ROOT}" \
  --out-summary "${OUT_SUMMARY}" \
  --out-subpops "${OUT_SUBPOPS}"

echo "[DONE $(date)]"

