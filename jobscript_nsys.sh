#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_nsys_%A.out
#SBATCH --error=logs/slurm_nsys_%A.err
#SBATCH --gres=gpu:l40s:1

# nsys profile of one proximal-lsq-auglag iteration of free.py.
# Perfetto / jax.profiler is intentionally not used (2 GiB XSpace overflow).
#
# Knobs (sbatch --export=ALL,NSYS_DELAY=900,NSYS_DURATION=480 ...):
#   NSYS_DELAY      seconds to skip before collection   (default 900)
#   NSYS_DURATION   seconds to collect                  (default 480)
#   NSYS_NO_GRAPHS  1 = disable XLA CUDA graphs         (default 1)
#
# Delay should land after "Starting optimization" (proximal build was
# 12.3 min in 16805595). Duration should cover one ~7 min iteration.
# If the first captured window is still compile, raise NSYS_DELAY.

mkdir -p logs nsys

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate desc

# nsys lives in the CUDA / Nsight modules on this cluster; try common names.
if ! command -v nsys >/dev/null 2>&1; then
  module load nsight-systems 2>/dev/null || \
  module load cuda/12.4 2>/dev/null || \
  module load cuda 2>/dev/null || true
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not on PATH. module avail nsight / cuda, then resubmit." >&2
  exit 1
fi

export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# NOTE: do NOT set JAX_PLATFORMS. coil_fem/magnetic.py repairs simsopt's
# jax_platform_name='cpu' pin only when JAX_PLATFORMS is unset.

# CUDA graphs collapse many XLA kernels into one graphLaunch, which makes
# the nsys timeline useless for "what is the 7 min". Default is off.
if [[ "${NSYS_NO_GRAPHS:-1}" == "1" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
  echo "XLA CUDA graphs disabled for a readable nsys timeline"
fi

NSYS_DELAY="${NSYS_DELAY:-900}"
NSYS_DURATION="${NSYS_DURATION:-480}"
REPORT="nsys/free_${SLURM_JOB_ID}"

echo "Job ID:         $SLURM_JOB_ID"
echo "nsys:           $(command -v nsys)  $(nsys --version | head -1)"
echo "NSYS_DELAY:     ${NSYS_DELAY}s"
echo "NSYS_DURATION:  ${NSYS_DURATION}s"
echo "XLA_FLAGS:      ${XLA_FLAGS:-<unset>}"
echo "Start time:     $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p data

# cuda + nvtx + the libraries XLA actually calls. Skip osrt: every malloc
# lands in the report and the file grows like the old Perfetto dump.
# --kill=none: when the duration window ends, leave python running so the
# report is flushed; the 2 h wall time then finishes the leftover iter.
nsys profile \
  --output="$REPORT" \
  --force-overwrite=true \
  --trace=cuda,nvtx,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --delay="$NSYS_DELAY" \
  --duration="$NSYS_DURATION" \
  --kill=none \
  python -u ./free.py
RC=$?

echo "----- nsys kernel summary (top 30 by time) -----"
nsys stats --force-export=true --report cuda_gpu_kern_sum "$REPORT.nsys-rep" \
  | head -50

echo "Report:         ${REPORT}.nsys-rep"
echo "Exit code:      $RC"
echo "End time:       $(date)"
exit $RC
