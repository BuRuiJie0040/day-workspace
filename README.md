# day

This repository now serves two purposes:

- a daily Git workspace for Codex-based development
- an academic-style MNIST handwritten digit recognition project

## Git and GitHub state

- Default branch: `main`
- Remote repository: `https://github.com/BuRuiJie0040/day-workspace.git`
- Git worktree support: verified locally
- Codex GitHub app installation: confirmed for `BuRuiJie0040`

For a fuller explanation of branches, worktrees, and GitHub integration, see
[docs/git-worktree-and-github-guide.md](D:\CODEX\day\docs\git-worktree-and-github-guide.md).

## MNIST project

The project trains a compact CNN with:

- manual MNIST downloader and IDX parser
- reproducible train/validation split
- AdamW optimizer and cosine learning-rate schedule
- checkpointing, metrics export, and visualization outputs

### Main files

- `train_mnist.py`: training entrypoint
- `mnist_project/data.py`: MNIST download and dataset loading
- `mnist_project/model.py`: CNN architecture
- `mnist_project/training.py`: training, evaluation, and artifact generation

### Run locally

```powershell
$env:HTTP_PROXY='http://127.0.0.1:7890'
$env:HTTPS_PROXY='http://127.0.0.1:7890'
D:\Anaconda\python.exe train_mnist.py --epochs 5 --batch-size 128 --cpu
```

### Quick verification run used here

```powershell
$env:HTTP_PROXY='http://127.0.0.1:7890'
$env:HTTPS_PROXY='http://127.0.0.1:7890'
D:\Anaconda\python.exe train_mnist.py --epochs 1 --batch-size 128 --train-subset 12000 --test-subset 2000 --cpu
```

That verification run produced:

- validation accuracy: `0.9692`
- test accuracy: `0.9545`
- macro F1: `0.9536`

Generated outputs are written to:

- `artifacts/metrics.json`
- `artifacts/classification_report.json`
- `artifacts/training_curves.png`
- `artifacts/confusion_matrix.png`
- `artifacts/sample_predictions.png`
