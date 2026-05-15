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

The project now outputs a complete local experiment package for MNIST:

- `best_model.pth`
- `Device.txt`
- `training_log_时间戳.log`
- `accuracy_comparison.png`
- `confusion_matrix.png`
- `tsne_points.png`
- `metrics.csv`
- `tsne_points.csv`

### Main files

- `train_mnist.py`: training entrypoint
- `mnist_project/data.py`: MNIST download and dataset loading
- `mnist_project/model.py`: CNN architecture
- `mnist_project/training.py`: training loop, logging, evaluation, and artifact generation

### Run locally

```powershell
D:\Anaconda\python.exe train_mnist.py --epochs 50 --batch-size 128 --cpu
```

This command does not require proxy environment variables by default. If your own
network cannot directly download MNIST, you can either:

- manually place the four MNIST IDX files under `data/raw`
- or configure your own proxy before running

### Quick verification run used here

```powershell
D:\Anaconda\python.exe train_mnist.py --epochs 3 --batch-size 128 --train-subset 12000 --test-subset 2000 --tsne-samples 300 --cpu
```

Generated outputs are written to:

- `artifacts/best_model.pth`
- `artifacts/Device.txt`
- `artifacts/training_log_*.log`
- `artifacts/metrics.csv`
- `artifacts/confusion_matrix.png`
- `artifacts/accuracy_comparison.png`
- `artifacts/tsne_points.png`
- `artifacts/tsne_points.csv`
