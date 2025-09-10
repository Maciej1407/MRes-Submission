#!/bin/bash -l
#$ -cwd

#$ -l h_rt=48:00:00
#$ -l mem=32G
#$ -o FinalFinal$JOB_ID.out
#$ -e FinalFinal$JOB_ID.err
#$ -N Final

# Read arguments
GRID_SIZE=$1
N_CELLS=$2
ENV=$3
LABEL=$4
ROOT="Final_S${GRID_SIZE}C${N_CELLS}_${ENV}/Maze"
# Initialize environment
source /etc/profile.d/modules.sh
module load python/3.9
source mvenv_py39/bin/activate

# Run metrics processor
echo "[START $(date)] Running metrics_processor.py on ${ROOT}"
python3 metrics_processor.py "${ROOT}" "${GRID_SIZE}" "${N_CELLS}" "${LABEL}"
echo "[DONE $(date)]"

