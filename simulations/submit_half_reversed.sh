#!/bin/bash
#SBATCH -J half_reversed
#SBATCH -A TG-DMR160007
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 04:00:00

#SBATCH -o simulations/logs/half_reversed_%j.out
#SBATCH -e simulations/logs/half_reversed_%j.err

set -euo pipefail

cd "$SCRATCH/projects/h-LFO-point-defects"

source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate emenv

DEVICE="gpu"
NUM_FP=200
NUM_GPUS=4
RUN_NAME="half_reversed_fp"

PRISTINE_FILE="data/structures/pristine.vasp"
REVERSED_FILE="data/structures/pristine_reversed.vasp"
X_REPEAT=1
Y_REPEAT=1
PRISTINE_LAYERS=12
REVERSED_LAYERS=13

SAVE_DIR="data/simulations/${RUN_NAME}"

mkdir -p "$SAVE_DIR"
mkdir -p simulations/logs

for STRUCTURE_FILE in "$PRISTINE_FILE" "$REVERSED_FILE"; do
  if [[ ! -f "$STRUCTURE_FILE" ]]; then
    echo "Structure file not found: ${STRUCTURE_FILE}" >&2
    exit 1
  fi
done

STRUCTURE_BLOCK_ARGS=(
  --structure-block "${PRISTINE_FILE},${X_REPEAT},${Y_REPEAT},${PRISTINE_LAYERS}"
  --structure-block "${REVERSED_FILE},${X_REPEAT},${Y_REPEAT},${REVERSED_LAYERS}"
)

export CUPY_GPU_MEMORY_LIMIT=$((75 * 1024**3))

PIDS=()

for ((GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++)); do
  (
    for ((FP_ID=GPU_ID; FP_ID<NUM_FP; FP_ID+=NUM_GPUS)); do
      SEED=$((1000 + FP_ID))

      echo "Starting frozen phonon ${FP_ID} on GPU ${GPU_ID} with seed ${SEED}"

      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      python simulations/run_simulation.py \
        --iter "$FP_ID" \
        --device "$DEVICE" \
        "${STRUCTURE_BLOCK_ARGS[@]}" \
        --save-dir "$SAVE_DIR" \
        --output-filename "${RUN_NAME}_${FP_ID}" \
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
        > "simulations/logs/${RUN_NAME}_${FP_ID}_${SLURM_JOB_ID}.out" \
        2> "simulations/logs/${RUN_NAME}_${FP_ID}_${SLURM_JOB_ID}.err"
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
  echo "At least one half-reversed frozen-phonon simulation failed." >&2
  exit 1
fi

echo "All ${NUM_FP} frozen-phonon simulations for ${RUN_NAME} completed."
