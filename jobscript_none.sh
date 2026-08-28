#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --gres=gpu:l40s:1


mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
conda activate desc
mkdir -p data_none
python -u ./none.py 

echo "End time:       $(date)"
