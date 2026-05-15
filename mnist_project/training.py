import csv
import json
import os
import platform
import random
import socket
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mnist_project.data import MNISTDataset
from mnist_project.model import AcademicMNISTNet


class ExperimentLogger(object):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(data_dir: Path, batch_size: int, train_subset: int = None, test_subset: int = None):
    train_full = MNISTDataset(data_dir, "train")
    test_full = MNISTDataset(data_dir, "test")

    if train_subset:
        train_full = Subset(train_full, list(range(min(train_subset, len(train_full)))))
    if test_subset:
        test_full = Subset(test_full, list(range(min(test_subset, len(test_full)))))

    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, epoch: int, epochs: int, stage: str):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(
        loader,
        desc="Epoch {}/{} [{}]".format(epoch, epochs, stage),
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for inputs, labels in progress:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        progress.set_postfix(loss="{:.4f}".format(total_loss / total), acc="{:.2f}%".format(100.0 * correct / total))
    progress.close()
    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(model, loader, device, collect_features: bool = False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits, features = model(inputs, return_features=True)
        probs = torch.softmax(logits, dim=1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_preds.append(probs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
        if collect_features:
            all_features.append(features.cpu())
    predictions = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probabilities = torch.cat(all_probs).numpy()
    result = {
        "predictions": predictions,
        "labels": labels,
        "probabilities": probabilities,
        "loss": total_loss / total,
        "accuracy": float((predictions == labels).mean()),
    }
    if collect_features:
        result["features"] = torch.cat(all_features).numpy()
    return result


def build_classification_report(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(10))
    )
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro")
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    report = {}
    for idx in range(10):
        report[str(idx)] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1-score": float(f1[idx]),
            "support": int(support[idx]),
        }
    report["macro avg"] = {
        "precision": float(macro[0]),
        "recall": float(macro[1]),
        "f1-score": float(macro[2]),
        "support": int(sum(support)),
    }
    report["weighted avg"] = {
        "precision": float(weighted[0]),
        "recall": float(weighted[1]),
        "f1-score": float(weighted[2]),
        "support": int(sum(support)),
    }
    report["accuracy"] = float((y_true == y_pred).mean())
    return report


def summarize_weight_discretization(model):
    all_errors = []
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.ndim < 2:
            continue
        weights = parameter.detach().cpu().numpy()
        max_abs = np.max(np.abs(weights))
        if max_abs == 0:
            continue
        scale = max_abs / 127.0
        discrete = np.round(weights / scale) * scale
        all_errors.append(np.abs(weights - discrete).ravel())
    if not all_errors:
        return {"mean_abs_error": 0.0, "max_abs_error": 0.0}
    errors = np.concatenate(all_errors)
    return {"mean_abs_error": float(errors.mean()), "max_abs_error": float(errors.max())}


def write_device_info(path: Path, device: torch.device) -> None:
    lines = [
        "Hostname: {}".format(socket.gethostname()),
        "Platform: {}".format(platform.platform()),
        "System: {} {}".format(platform.system(), platform.release()),
        "Machine: {}".format(platform.machine()),
        "Processor: {}".format(platform.processor()),
        "Python: {}".format(sys.version.replace("\n", " ")),
        "PyTorch: {}".format(torch.__version__),
        "CUDA available: {}".format(torch.cuda.is_available()),
        "Selected device: {}".format(device),
        "CPU count: {}".format(os.cpu_count()),
    ]
    if torch.cuda.is_available():
        lines.append("CUDA device count: {}".format(torch.cuda.device_count()))
        lines.append("CUDA device name: {}".format(torch.cuda.get_device_name(0)))
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def write_metrics_csv(history_rows, path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "lr",
        "epoch_time_seconds",
        "quant_mean_abs_error",
        "quant_max_abs_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


def plot_accuracy_comparison(history_rows, save_path: Path) -> None:
    epochs = [row["epoch"] for row in history_rows]
    train_acc = [row["train_acc"] * 100.0 for row in history_rows]
    test_acc = [row["test_acc"] * 100.0 for row in history_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, test_acc, marker="s", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=180)
    plt.close()


def plot_confusion_matrix(confusion, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(10)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    threshold = confusion.max() / 2.0 if confusion.max() else 0.0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > threshold else "black",
            )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=180)
    plt.close()


def create_tsne_artifacts(features, labels, predictions, save_png: Path, save_csv: Path):
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(features)
    with save_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "label", "prediction"])
        for point, label, pred in zip(embedding, labels, predictions):
            writer.writerow([float(point[0]), float(point[1]), int(label), int(pred)])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
    plt.title("t-SNE Feature Projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    handles = []
    for digit in range(10):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(digit),
                markerfacecolor=plt.cm.tab10(float(digit) / 10.0),
                markersize=6,
            )
        )
    plt.legend(handles=handles, title="Digit", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(save_png), dpi=180)
    plt.close()


def train_model(args):
    set_seed(args.seed)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = ExperimentLogger(output_dir / "training_log_{}.log".format(timestamp))
    best_model_path = output_dir / "best_model.pth"
    device_info_path = output_dir / "Device.txt"
    metrics_csv_path = output_dir / "metrics.csv"
    confusion_png_path = output_dir / "confusion_matrix.png"
    accuracy_png_path = output_dir / "accuracy_comparison.png"
    tsne_png_path = output_dir / "tsne_points.png"
    tsne_csv_path = output_dir / "tsne_points.csv"
    metrics_json_path = output_dir / "metrics.json"
    report_json_path = output_dir / "classification_report.json"

    write_device_info(device_info_path, device)
    logger.log("-> 实验启动，设备信息已写入 {}".format(device_info_path.name))

    train_loader, test_loader = prepare_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    model = AcademicMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history_rows = []
    best_test_acc = -1.0

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, "Train")
            logger.log("-> Epoch {} 训练完成，正在进行权重离散化映射..".format(epoch))
            quant_stats = summarize_weight_discretization(model)
            logger.log(
                "-> Epoch {} 离散化映射完成 | MeanAbsError: {:.6f} | MaxAbsError: {:.6f}".format(
                    epoch,
                    quant_stats["mean_abs_error"],
                    quant_stats["max_abs_error"],
                )
            )

            test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, epoch, args.epochs, "Test")
            epoch_time = time.time() - epoch_start
            scheduler.step()

            history_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(test_acc, 6),
                "lr": round(current_lr, 8),
                "epoch_time_seconds": round(epoch_time, 3),
                "quant_mean_abs_error": round(quant_stats["mean_abs_error"], 8),
                "quant_max_abs_error": round(quant_stats["max_abs_error"], 8),
            }
            history_rows.append(history_row)

            logger.log(
                "Ep {}/{} | TrAcc: {:.2f}% | TeAcc: {:.2f}% | LR: {:.5f} | Time: {:.1f}s".format(
                    epoch,
                    args.epochs,
                    train_acc * 100.0,
                    test_acc * 100.0,
                    current_lr,
                    epoch_time,
                )
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), best_model_path)
                logger.log("-> [NEW BEST] Model saved with Test Accuracy: {:.2f}%".format(test_acc * 100.0))

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        evaluation = collect_predictions(model, test_loader, device, collect_features=True)
        report = build_classification_report(evaluation["labels"], evaluation["predictions"])
        conf = confusion_matrix(evaluation["labels"], evaluation["predictions"])

        write_metrics_csv(history_rows, metrics_csv_path)
        plot_accuracy_comparison(history_rows, accuracy_png_path)
        plot_confusion_matrix(conf, confusion_png_path)

        tsne_count = min(args.tsne_samples, len(evaluation["labels"]))
        create_tsne_artifacts(
            evaluation["features"][:tsne_count],
            evaluation["labels"][:tsne_count],
            evaluation["predictions"][:tsne_count],
            tsne_png_path,
            tsne_csv_path,
        )

        metrics_summary = {
            "best_test_acc": float(best_test_acc),
            "final_test_acc": float(evaluation["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "device": str(device),
            "outputs": {
                "best_model": best_model_path.name,
                "device_info": device_info_path.name,
                "training_log": logger.log_path.name,
                "accuracy_comparison": accuracy_png_path.name,
                "confusion_matrix": confusion_png_path.name,
                "tsne_points": tsne_png_path.name,
                "metrics_csv": metrics_csv_path.name,
                "tsne_csv": tsne_csv_path.name,
            },
        }
        with metrics_json_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_summary, file, indent=2)
        with report_json_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        logger.log("-> 训练完成，核心产物已输出到 {}".format(output_dir))
        logger.log(json.dumps(metrics_summary, indent=2))
    finally:
        logger.close()
