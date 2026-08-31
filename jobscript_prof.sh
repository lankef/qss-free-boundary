#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm_mem_%A.out
#SBATCH --error=logs/slurm_mem_%A.err
#SBATCH --gres=gpu:l40s:1

# Memory-diagnostic run. No nsys: the profiler adds overhead and its
# "Memory Usage" row reports cumulative driver-level reservation, which is
# the number we are trying to go behind. Here we measure the JAX allocator
# from inside the process and the true device total from outside it.
#
# Toggles (set on the sbatch command line, e.g. `sbatch --export=ALL,DUMP_HLO=1 ...`):
#   DUMP_HLO=1      dump HLO + buffer assignment   -> step 2, only if JAX peak dominates
#   NO_AUTOTUNE=1   disable XLA autotuning         -> tests whether the peak is compile-time scratch
# To filter out memory requests for analysis with an ai, run something like this
# tar -cvf hlo_dump_15483668.tar  hlo_dump_15483668/*buffer-assignment.txt   hlo_dump_15483668/*after_optimizations.txt   hlo_dump_15483668/*.hlo_module_config* 2>/dev/null

mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate desc

export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# NOTE: do NOT set JAX_PLATFORMS. coil_fem/magnetic.py repairs simsopt's
# jax_platform_name='cpu' pin only when JAX_PLATFORMS is unset.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
XLA_FLAGS_ACC=""

if [[ "${DUMP_HLO:-0}" == "1" ]]; then
  DUMP_DIR="${SUBMIT_DIR}/hlo_dump_${SLURM_JOB_ID}"
  mkdir -p "$DUMP_DIR"
  # No --xla_dump_hlo_pass_re: that dumps after every pass (hundreds of files,
  # GBs, much slower compile). The default dump already includes the
  # *buffer-assignment.txt files, which are what matter for memory.
  XLA_FLAGS_ACC="${XLA_FLAGS_ACC} --xla_dump_to=${DUMP_DIR}"
  echo "HLO dump enabled: $DUMP_DIR"
fi

if [[ "${NO_AUTOTUNE:-0}" == "1" ]]; then
  XLA_FLAGS_ACC="${XLA_FLAGS_ACC} --xla_gpu_autotune_level=0"
  echo "XLA autotuning disabled"
fi

[[ -n "$XLA_FLAGS_ACC" ]] && export XLA_FLAGS="$XLA_FLAGS_ACC"

# REMAT_INT="${SLURM_ARRAY_TASK_ID:-0}"
# MEMCSV="${SUBMIT_DIR}/logs/mem_${SLURM_JOB_ID}_remat${REMAT_INT}.csv"

echo "Job ID:         $SLURM_JOB_ID"
# echo "Array task ID:  ${SLURM_ARRAY_TASK_ID:-<none>}"
# echo "remat_int:      $REMAT_INT"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "XLA_FLAGS:      ${XLA_FLAGS:-<unset>}"
echo "Start time:     $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- external sampler: sees ALL device memory, including cuDSS ----
nvidia-smi --query-gpu=timestamp,memory.used,memory.total \
           --format=csv,noheader,nounits -lms 200 > "$MEMCSV" &
SMI_PID=$!
# Make sure the sampler dies with the job, however the job ends.
trap 'kill $SMI_PID 2>/dev/null' EXIT

python -u ./free.py # "$REMAT_INT"
RC=$?

sleep 1
kill $SMI_PID 2>/dev/null
wait $SMI_PID 2>/dev/null

echo "----- device memory (external sampler) -----"
python - "$MEMCSV" <<'EOF'
import sys
rows = []
for line in open(sys.argv[1]):
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 3:
        continue
    try:
        rows.append((parts[0], int(parts[1]), int(parts[2])))
    except ValueError:
        continue
if not rows:
    print("no samples collected")
    sys.exit()
used = [r[1] for r in rows]
peak_i = used.index(max(used))
print(f"samples:        {len(rows)}")
print(f"baseline used:  {used[0]/1024:.3f} GiB")
print(f"PEAK used:      {max(used)/1024:.3f} GiB   at {rows[peak_i][0]}")
print(f"final used:     {used[-1]/1024:.3f} GiB")
print(f"device total:   {rows[0][2]/1024:.3f} GiB")
EOF

if [[ "${DUMP_HLO:-0}" == "1" ]]; then
  echo "----- XLA buffer assignment (largest modules) -----"
  for f in "${SUBMIT_DIR}/hlo_dump_${SLURM_JOB_ID}"/*buffer-assignment.txt; do
    [[ -e "$f" ]] || { echo "no buffer-assignment files produced"; break; }
    echo "== $(basename "$f")"
    grep -E "Total bytes used|^\s*allocation [0-9]+:" "$f" | head -25
  done
fi

echo "Exit code:      $RC"
echo "End time:       $(date)"
exit $RC

