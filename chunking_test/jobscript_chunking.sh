#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_chunk_%A.out
#SBATCH --error=logs/slurm_chunk_%A.err
#SBATCH --gres=gpu:l40s:1

# Accuracy + memory check for quadcoil's jac_chunk_size on an L40S.
#
# The local dev GPU is too small to run this, and the answer is
# device-dependent: on CPU the chunking error was ~1e-10 relative, but a GPU
# run of the same case disagreed by ~40%. So it has to be measured here.
#
# The python driver spawns one subprocess per configuration so that the
# reported peak device memory belongs to that configuration alone (JAX's
# allocator peak counter is a running maximum and never resets).
#
# Knobs (set on the sbatch command line, e.g.
#   sbatch --export=ALL,MPOL=10,NTOR=10 jobscript_chunking.sh):
#   MPOL, NTOR      current-potential resolution      (default 4, 4)
#   NPHI, NTHETA    quadrature points per period      (default 8, 8)
#   MAXITER         solver iterations                 (default 200)
#   CHUNKS          comma-separated sweep list        (default 1,5,8,20,40)
#   REG             f_K weights, 0 = unregularized    (default 0,1e-14)
#   QUADCOIL_TESTS  path to quadcoil's tests/ dir (needs surfaces.json)
#
# Note that a jac_chunk_size larger than the number of metric rows is the
# same as no chunking at all, so keep CHUNKS below ndofs for the chosen
# mpol/ntor (40 rows at mpol=ntor=4, 220 at mpol=ntor=10).

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
export OUTDIR="${OUTDIR:-${SUBMIT_DIR}/out_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUTDIR"

echo "Job ID:         ${SLURM_JOB_ID:-<none>}"
echo "CPUs per task:  ${SLURM_CPUS_PER_TASK:-<none>}"
echo "Output dir:     $OUTDIR"
echo "Start time:     $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- external sampler: sees ALL device memory, including cuDSS ----
# The in-process JAX numbers miss anything allocated outside the JAX
# allocator, so keep an independent view of the device total.
MEMCSV="${OUTDIR}/mem_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total \
           --format=csv,noheader,nounits -lms 200 > "$MEMCSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

python -u "${SUBMIT_DIR}/test_chunking.py"
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

echo "Exit code:      $RC"
echo "End time:       $(date)"
exit $RC
