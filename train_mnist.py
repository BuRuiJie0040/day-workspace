import argparse

from mnist_project.training import train_model


def build_parser():
    parser = argparse.ArgumentParser(description="Train an academic-style MNIST classifier.")
    parser.add_argument("--data-dir", default="data", help="Directory for downloaded MNIST files.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for generated reports and figures.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for model checkpoints.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio from training set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-subset", type=int, default=None, help="Optional cap on training examples.")
    parser.add_argument("--test-subset", type=int, default=None, help="Optional cap on test examples.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_model(args)
