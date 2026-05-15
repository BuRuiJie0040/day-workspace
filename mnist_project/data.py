import gzip
import shutil
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MNIST_RESOURCES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]


def download_mnist(data_dir: Path) -> None:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_RESOURCES.values():
        gz_path = raw_dir / filename
        extracted_path = raw_dir / filename.replace(".gz", "")
        if extracted_path.exists():
            continue
        if not gz_path.exists():
            last_error = None
            for mirror in MNIST_MIRRORS:
                try:
                    urllib.request.urlretrieve(mirror + filename, gz_path.as_posix())
                    last_error = None
                    break
                except Exception as exc:  # pragma: no cover - network errors are environment-specific.
                    last_error = exc
            if last_error is not None:
                raise RuntimeError(f"Failed to download {filename}: {last_error}")
        with gzip.open(gz_path, "rb") as src, extracted_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number for images: {magic}")
        buffer = f.read()
    images = np.frombuffer(buffer, dtype=np.uint8).reshape(num_images, rows, cols)
    return images


def _read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number for labels: {magic}")
        buffer = f.read()
    labels = np.frombuffer(buffer, dtype=np.uint8).reshape(num_items)
    return labels


class MNISTDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        download_mnist(data_dir)
        raw_dir = data_dir / "raw"
        image_key = "train_images" if split == "train" else "test_images"
        label_key = "train_labels" if split == "train" else "test_labels"
        images = _read_idx_images(raw_dir / MNIST_RESOURCES[image_key].replace(".gz", ""))
        labels = _read_idx_labels(raw_dir / MNIST_RESOURCES[label_key].replace(".gz", ""))
        self.images = torch.from_numpy(images.copy()).float().unsqueeze(1) / 255.0
        self.labels = torch.from_numpy(labels.copy()).long()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]
