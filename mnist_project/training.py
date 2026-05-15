import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from mnist_project.data import MNISTDataset
from mnist_project.model import AcademicMNISTNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(data_dir: Path, batch_size: int, val_ratio: float, seed: int,
                        train_subset: int = None, test_subset: int = None):
    train_full = MNISTDataset(data_dir, "train")
    test_full = MNISTDataset(data_dir, "test")

    if train_subset:
        train_full = Subset(train_full, list(range(min(train_subset, len(train_full)))))
    if test_subset:
        test_full = Subset(test_full, list(range(min(test_subset, len(test_full)))))

    val_size = max(1, int(len(train_full) * val_ratio))
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels_all = []
    probabilities = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())
        probabilities.append(probs.cpu())
    return (
        torch.cat(predictions).numpy(),
        torch.cat(labels_all).numpy(),
        torch.cat(probabilities).numpy(),
    )


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = prepare_dataloaders(
        Path(args.data_dir),
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    model = AcademicMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_acc = -math.inf
    best_checkpoint = checkpoint_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_checkpoint,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_pred, y_true, probabilities = evaluate(model, test_loader, device)

    conf = confusion_matrix(y_true, y_pred)
    report = build_classification_report(y_true, y_pred)
    metrics = {
        "device": str(device),
        "best_val_acc": float(best_val_acc),
        "test_accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "history": history,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_path = output_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_matrix(conf, output_dir / "confusion_matrix.png")
    plot_sample_predictions(test_loader, y_pred, y_true, probabilities, output_dir / "sample_predictions.png")

    print(json.dumps({
        "best_val_acc": round(best_val_acc, 4),
        "test_accuracy": round(metrics["test_accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "artifacts": str(output_dir),
    }, indent=2))


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


def plot_training_curves(history, save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(list(epochs), history["train_loss"], label="train")
    plt.plot(list(epochs), history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list(epochs), history["train_acc"], label="train")
    plt.plot(list(epochs), history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160)
    plt.close()


def plot_confusion_matrix(confusion, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.title("MNIST confusion matrix")
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
    plt.savefig(str(save_path), dpi=160)
    plt.close()


def plot_sample_predictions(loader, y_pred, y_true, probabilities, save_path: Path) -> None:
    images = []
    labels = []
    for batch_images, batch_labels in loader:
        for image, label in zip(batch_images, batch_labels):
            images.append(image.squeeze(0).numpy())
            labels.append(int(label))
            if len(images) == 16:
                break
        if len(images) == 16:
            break

    plt.figure(figsize=(10, 10))
    for idx, image in enumerate(images):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(image, cmap="gray")
        confidence = float(probabilities[idx, y_pred[idx]])
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]} ({confidence:.2f})", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160)
    plt.close()
