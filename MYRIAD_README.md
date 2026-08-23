# UCL Myriad deployment

Do not submit the 95-task array until the one-task GPU benchmark succeeds.

## 1. After the account-created email

Log in and create directories:

```bash
ssh YOUR_UCL_ID@myriad.rc.ucl.ac.uk
mkdir -p "$HOME/Scratch/dissertation_rl/logs" "$HOME/venvs"
```

Use `scp` or SFTP to upload this project to
`$HOME/Scratch/dissertation_rl`. Do not upload old checkpoint, benchmark, or
smoke-test directories unless they are needed for archival purposes.

## 2. Inspect current modules

On the login node run:

```bash
module avail python
module avail cuda
module avail pytorch
```

The available module versions change. Select mutually compatible Python,
CUDA, and CUDA-enabled PyTorch versions; do not copy an obsolete version from
an old example. If using a UCL PyTorch module, create the virtual environment
with `--system-site-packages` so it can see that module.

```bash
python -m venv --system-site-packages "$HOME/venvs/dissertation-rl"
source "$HOME/venvs/dissertation-rl/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-myriad.txt
```

Before submitting anything, verify imports. GPU availability should be tested
inside an allocated GPU job, not by training on the shared login node.

## 3. GPU benchmark

Update the module-load block in both `.sh` files to match the modules used to
create the environment. Then:

```bash
chmod +x myriad_gpu_benchmark.sh myriad_gpu_array.sh
mkdir -p logs
qsub myriad_gpu_benchmark.sh
qstat
```

Review the `.o` and `.e` files in `logs`. The benchmark must report
`torch.cuda.is_available() == True`, an NVIDIA GPU name, and a completed JSON
task result.

## 4. Formal array

Only after checking benchmark speed and outputs:

```bash
qsub myriad_gpu_array.sh
qstat
```

The array is capped at four concurrent tasks (`#$ -tc 4`). Each completed
task writes an atomic JSON marker. Re-running a task without `--retrain`
skips an already completed result, and compatible checkpoints are reused.

After the array finishes, audit and aggregate all annual paths:

```bash
python collect_walkforward_results.py --project-root "$HOME/Scratch/dissertation_rl"
```

The command exits non-zero if any of the 95 tasks are missing. It produces
annual-fold results, genuine pooled 2021-2025 metrics for each seed, and the
model-level mean/standard deviation workbook.

Myriad login nodes are for file management, environment setup, and short
tests only. Submit training through `qsub`.
