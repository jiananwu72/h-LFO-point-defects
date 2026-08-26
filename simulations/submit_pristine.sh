#!/bin/bash
#SBATCH -J pristine_simulation
#SBATCH -A TG-DMR160007
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 04:00:00

#SBATCH -o simulations/logs/simulation_%j.out
#SBATCH -e simulations/logs/simulation_%j.err

set -euo pipefail

cd "$SCRATCH/projects/h-LFO-point-defects"

# Activate Conda environment.
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate emenv

DEVICE=gpu

NUM_FP=200
NUM_GPUS=4
RUN_NAME="pristine_fp"

FILE_1="data/structures/pristine.vasp"
X_1=1
Y_1=1
LAYERS_1=25

STRUCTURE_BLOCKS=(
  "${FILE_1},${X_1},${Y_1},${LAYERS_1}"
)

SAVE_DIR="data/simulations/${RUN_NAME}"

mkdir -p "$SAVE_DIR"
mkdir -p simulations/logs

STRUCTURE_BLOCK_ARGS=()
for BLOCK in "${STRUCTURE_BLOCKS[@]}"; do
  STRUCTURE_BLOCK_ARGS+=(--structure-block "$BLOCK")
done

# Limit applies independently inside each GPU-bound Python process.
export CUPY_GPU_MEMORY_LIMIT=$((75 * 1024**3))

PIDS=()

# Start one worker process per GPU.
for ((GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++)); do
  (
    # Each GPU runs every NUM_GPUS-th frozen-phonon simulation.
    for ((FP_ID=GPU_ID+100; FP_ID<NUM_FP; FP_ID+=NUM_GPUS)); do
      SEED=$((1000 + FP_ID))

      echo "Starting frozen phonon $FP_ID on GPU $GPU_ID with seed $SEED"

      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      python simulations/run_simulation.py \
        --iter "$FP_ID" \
        --device "$DEVICE" \
        "${STRUCTURE_BLOCK_ARGS[@]}" \
        --save-dir "$SAVE_DIR" \
        --output-filename "${RUN_NAME}_fp_${FP_ID}" \
        --energy 100000 \
        --cs 0 \
        --semiangle-cutoff 30 \
        --defocus 10 \
        --no-ensemble-mean \
        --frozen-phonon-configs 1 \
        --frozen-phonon-seed "$SEED" \
        --frozen-phonon-sigma 0.07 \
        --potential-sampling 0.05 \
        --scan-start "0.083333,0" \
        --scan-end "0.916667,1" \
        --detector-inner 80 \
        --detector-outer 200 \
        --interpolation-sampling 0.05 \
        --dose-per-area 1e7 \
        --max-batch "1 GB" \
        > "simulations/logs/${RUN_NAME}_fp_${FP_ID}_${SLURM_JOB_ID}.out" \
        2> "simulations/logs/${RUN_NAME}_fp_${FP_ID}_${SLURM_JOB_ID}.err"
    done
  ) &

  PIDS+=("$!")
done

FAILED=0

for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
  fi
done

if (( FAILED != 0 )); then
  echo "At least one frozen-phonon simulation failed." >&2
  exit 1
fi

echo "All ${NUM_FP} frozen-phonon simulations for ${RUN_NAME} completed."