#!/bin/bash
#SBATCH -J defect_simulation
#SBATCH -A TG-DMR160007
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH --array=0-3

#SBATCH -o simulations/logs/defect_%A_%a.out
#SBATCH -e simulations/logs/defect_%A_%a.err

set -euo pipefail

cd "$SCRATCH/projects/h-LFO-point-defects"

source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate emenv

DEVICE="gpu"
NUM_ENSEMBLES=100
NUM_GPUS=4
NUM_ARRAY_TASKS=4
# Change this to the defect you want to simulate
DEFECT="LuFe_down"

PRISTINE_FILE="data/structures/pristine.vasp"
DEFECT_FILE="data/structures/${DEFECT}.vasp"

X_REPEAT=1
Y_REPEAT=1

mkdir -p simulations/logs
SAVE_ROOT="data/simulations/${DEFECT}"

if [[ ! -f "$DEFECT_FILE" ]]; then
  echo "Defect structure file not found: ${DEFECT_FILE}" >&2
  exit 1
fi

mkdir -p "$SAVE_ROOT"

export CUPY_GPU_MEMORY_LIMIT=$((75 * 1024**3))

# Array task 0 processes layers 0,4,8,12,16,20,24
# Array task 1 processes layers 1,5,9,13,17,21
# Array task 2 processes layers 2,6,10,14,18,22
# Array task 3 processes layers 3,7,11,15,19,23
for ((DEFECT_LAYER=SLURM_ARRAY_TASK_ID;
      DEFECT_LAYER<25;
      DEFECT_LAYER+=NUM_ARRAY_TASKS)); do

  PRISTINE_BEFORE="$DEFECT_LAYER"
  PRISTINE_AFTER=$((24 - DEFECT_LAYER))

  RUN_NAME="${DEFECT}_layer_${DEFECT_LAYER}"
  SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"

  mkdir -p "$SAVE_DIR"

  STRUCTURE_BLOCK_ARGS=(
    --structure-block "${PRISTINE_FILE},${X_REPEAT},${Y_REPEAT},${PRISTINE_BEFORE}"
    --structure-block "${DEFECT_FILE},${X_REPEAT},${Y_REPEAT},1"
    --structure-block "${PRISTINE_FILE},${X_REPEAT},${Y_REPEAT},${PRISTINE_AFTER}"
  )

  echo "Starting ${DEFECT} at layer ${DEFECT_LAYER}"

  PIDS=()

  for ((GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++)); do
    (
      for ((ENSEMBLE_ID=GPU_ID;
            ENSEMBLE_ID<NUM_ENSEMBLES;
            ENSEMBLE_ID+=NUM_GPUS)); do

        SEED=$((100000 + DEFECT_LAYER * NUM_ENSEMBLES + ENSEMBLE_ID))

        echo "Layer ${DEFECT_LAYER}, ensemble ${ENSEMBLE_ID}, GPU ${GPU_ID}, seed ${SEED}"

        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python simulations/run_simulation.py \
          --iter "$ENSEMBLE_ID" \
          --device "$DEVICE" \
          "${STRUCTURE_BLOCK_ARGS[@]}" \
          --save-dir "$SAVE_DIR" \
          --output-filename "${RUN_NAME}_ensemble_${ENSEMBLE_ID}" \
          --energy 100000 \
          --cs 0 \
          --semiangle-cutoff 30 \
          --defocus 10 \
          --no-ensemble-mean \
          --frozen-phonon-configs 1 \
          --frozen-phonon-seed "$SEED" \
          --frozen-phonon-sigma 0.07 \
          --potential-sampling 0.05 \
          --scan-start "0.25,0.16667" \
          --scan-end "0.75,0.83333" \
          --detector-inner 80 \
          --detector-outer 200 \
          --interpolation-sampling 0.05 \
          --dose-per-area 1e7 \
          --max-batch "1 GB" \
          > "simulations/logs/${RUN_NAME}_ensemble_${ENSEMBLE_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
          2> "simulations/logs/${RUN_NAME}_ensemble_${ENSEMBLE_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
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
    echo "At least one simulation failed for defect layer ${DEFECT_LAYER}." >&2
    exit 1
  fi

  echo "Completed all ${NUM_ENSEMBLES} ${DEFECT} ensembles for layer ${DEFECT_LAYER}."

done

echo "Array task ${SLURM_ARRAY_TASK_ID} completed all assigned defect layers."
