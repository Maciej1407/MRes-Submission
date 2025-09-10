#!/bin/bash -l
# Batch script to run each parameter combination as its own SGE array task

# 1) 48 h wallclock time
#$ -l h_rt=48:0:0
# 2) 32 GB RAM
#$ -l mem=32G
# 3) Job array size: 14 consumption pairs × 4 diffusion × 4 epochs = 224
#$ -t 1-224
# 4) Name your job
#$ -N Size500Cells50
# 5) Working directory
#$ -wd /home/dmcbmkm/Scratch/pra_run

module load python3/3.9

cd /home/dmcbmkm/Scratch/pra_run
source mvenv_py39/bin/activate

consumption_rate_pairs=(
  "0.0001,single" "0.001,single" "0.01,single" "0.1,single" "1.0,single"
  "0.0001,0"     "0.001,0"     "0.01,0"     "0.1,0"     "1,0"
  "0.0001,0.01"  "0.01,0.001"  "1,0.0001"   "0.5,0.1"
)
diffusion_rates=( 0.2 0.5 1 3 )
epochs=( 0 1 2 3 )

# Build a flat list of all combinations
idx=0
for cr in "${consumption_rate_pairs[@]}"; do
  for diff in "${diffusion_rates[@]}"; do
    for ep in "${epochs[@]}"; do
      combos[$idx]="$cr|$diff|$ep"
      (( idx++ ))
    done
  done
done

# Pick the combination for this array task
combo="${combos[$((SGE_TASK_ID-1))]}"
IFS='|' read -r cr_pair diff ep <<< "$combo"
IFS=',' read -r fR sR <<< "$cr_pair"


echo "$(date '+%Y-%m-%d %H:%M:%S') - TASK $SGE_TASK_ID → fR=${fR}, sR=${sR}, diff=${diff}, epoch=${ep}" \
    >> simulation_status_p.log

# Run one simulation with the refactored script 
python3 Simulation_macrophages_hpc.py \
  -env Maze \
  -fR   "$fR" \
  -sR   "$sR" \
  -diff "$diff" \
  -ep     "$ep"  \
  -Nx    500  \
  -Ny    500  \
  -cells  50 \
  -root SimulationResults_S500_C50

echo "Task $SGE_TASK_ID complete: Maze, fR=${fR}, sR=${sR}, diff=${diff}, epoch=${ep}"
