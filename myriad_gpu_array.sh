#!/bin/bash -l

# UCL Myriad SGE array template: one GPU and one formal experiment per task.
#$ -N rl_wf_v6
#$ -l h_rt=8:00:00
#$ -l mem=4G
#$ -l tmpfs=10G
#$ -l gpu=1
#$ -t 1-95
#$ -tc 4
#$ -cwd
#$ -o logs
#$ -e logs

set -euo pipefail

PROJECT_ROOT="${HOME}/Scratch/dissertation_rl"
VENV_DIR="${HOME}/venvs/dissertation-rl"

cd "$PROJECT_ROOT"

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python/3.9.6-gnu-10.2.0
module load cuda/11.8.0/gnu-10.2.0
module load cudnn/9.2.0.82/cuda-11
module load pytorch/2.1.0/gpu

source "${VENV_DIR}/bin/activate"
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

python -u run_walkforward_task.py \
  --manifest walkforward_manifest.tsv \
  --task-id "${SGE_TASK_ID}" \
  --project-root "$PROJECT_ROOT"
