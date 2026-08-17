#!/bin/bash
#SBATCH -J layered_simulation
#SBATCH -A TG-DMR160007
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00

#SBATCH -o simulations/logs/simulation_%j.out
#SBATCH -e simulations/logs/simulation_%j.err

set -euo pipefail
cd "$SCRATCH/projects/h-LFO-point-defects"

# Activate Conda environment.
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate emenv

DEVICE=gpu

FILE_1="data/structures/pristine.vasp"
X_1=5
Y_1=1
LAYERS_1=25

STRUCTURE_BLOCKS=(
  "${FILE_1},${X_1},${Y_1},${LAYERS_1}"
)

SAVE_DIR="data/simulations/test"

mkdir -p "$SAVE_DIR"
mkdir -p simulations/logs

STRUCTURE_BLOCK_ARGS=()
for BLOCK in "${STRUCTURE_BLOCKS[@]}"; do
  STRUCTURE_BLOCK_ARGS+=(--structure-block "$BLOCK")
done

# Limit applies independently inside each GPU-bound Python process.
export CUPY_GPU_MEMORY_LIMIT=$((75 * 1024**3))

PIDS=()

for GPU_ID in 0 1 2 3; do
  SEED=$((1000 + GPU_ID))

  echo "Starting frozen phonon $GPU_ID on GPU $GPU_ID with seed $SEED"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python simulations/run_simulation.py \
    --iter "$GPU_ID" \
    --device "$DEVICE" \
    "${STRUCTURE_BLOCK_ARGS[@]}" \
    --save-dir "$SAVE_DIR" \
    --output-filename "test_fp_${GPU_ID}" \
    --energy 100000 \
    --cs 0 \
    --semiangle-cutoff 30 \
    --defocus 10 \
    --no-ensemble-mean \
    --frozen-phonon-configs 1 \
    --frozen-phonon-seed "$SEED" \
    --frozen-phonon-sigma 0.07 \
    --potential-sampling 0.05 \
    --scan-start "0.25,0.2" \
    --scan-end "0.75,0.8" \
    --detector-inner 80 \
    --detector-outer 200 \
    --interpolation-sampling 0.05 \
    --dose-per-area 1e7 \
    --max-batch "1 GB" \
    > "simulations/logs/fp_${GPU_ID}_${SLURM_JOB_ID}.out" \
    2> "simulations/logs/fp_${GPU_ID}_${SLURM_JOB_ID}.err" &

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

echo "All frozen-phonon simulations completed."