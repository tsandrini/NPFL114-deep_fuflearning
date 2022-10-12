#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
import sys
import re
import random

import urllib.request
from typing import Tuple, Dict, Iterator, Optional

import numpy as np
import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--activation", default="none", type=str, help="Activation function."
)
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument(
    "--hidden_layer", default=100, type=int, help="Size of the hidden layer."
)
parser.add_argument("--hidden_layers", default=1, type=int, help="Number of layers.")
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# If you add more arguments, ReCodEx will keep them with your default values.


class MNIST:
    H: int = 28
    W: int = 28
    C: int = 1
    LABELS: int = 10

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/datasets/"

    class Dataset:
        def __init__(
            self, data: Dict[str, np.ndarray], shuffle_batches: bool, seed: int = 42
        ) -> None:
            self._data = data
            self._data["images"] = self._data["images"].astype(np.float32) / 255
            self._size = len(self._data["images"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self) -> Dict[str, np.ndarray]:
            return self._data

        @property
        def size(self) -> int:
            return self._size

        def batches(
            self, size: Optional[int] = None
        ) -> Iterator[Dict[str, np.ndarray]]:
            permutation = (
                self._shuffler.permutation(self._size)
                if self._shuffler
                else np.arange(self._size)
            )
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

        @property
        def dataset(self) -> TensorDataset:
            return TensorDataset(
                torch.Tensor(self._data["images"]),
                torch.Tensor(self._data["labels"]).type(torch.LongTensor),
            )
            # return tf.data.Dataset.from_tensor_slices(self._data)

    def __init__(self, dataset: str = "mnist", size: Dict[str, int] = {}) -> None:
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {
                key[len(dataset) + 1 :]: mnist[key][: size.get(dataset, None)]
                for key in mnist
                if key.startswith(dataset)
            }
            setattr(
                self, dataset, self.Dataset(data, shuffle_batches=dataset == "train")
            )

    train: Dataset
    dev: Dataset
    test: Dataset


ACTIVATION_FUNCS = {
    None: None,
    "none": None,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()

        def _linear_block(in_dim, out_dim, activation=None):
            layers = []
            layers.append(nn.Linear(in_dim, out_dim))
            if ACTIVATION_FUNCS[activation] is not None:
                layers.append(ACTIVATION_FUNCS[activation]())

            return layers

        layers = _linear_block(
            MNIST.H * MNIST.W * MNIST.C, args.hidden_layer, args.activation
        )
        for _ in range(args.hidden_layers - 1):
            layers.extend(
                _linear_block(args.hidden_layer, args.hidden_layer, args.activation)
            )

        self.ffw = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.ffw(x)


def train_loop(dataloader, model, loss_fn, optimizer, step=None, writer=None):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if writer is not None:
                writer.add_scalar("loss", loss, step * (current / size))


def test_loop(dataloader, model, loss_fn, step=None, writer=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Dev Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    if writer is not None:
        writer.add_scalar("val_loss", test_loss, step)
        writer.add_scalar("val_acc", 100 * correct, step)


def main(args: argparse.Namespace) -> float:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.set_num_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                (
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                    for k, v in sorted(vars(args).items())
                )
            ),
        ),
    )
    writer = SummaryWriter(args.logdir)

    mnist = MNIST()
    model = Model()

    train_dataloader = DataLoader(
        mnist.train.dataset, batch_size=args.batch_size, shuffle=True
    )
    dev_dataloader = DataLoader(
        mnist.dev.dataset, batch_size=args.batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        mnist.test.dataset, batch_size=args.batch_size, shuffle=False
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    x_sample, _ = iter(dev_dataloader).next()
    writer.add_graph(model, x_sample)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t, writer)
        test_loop(dev_dataloader, model, loss_fn, t, writer)
    print("Done!")

    print("Test metrics:")
    test_loop(test_dataloader, model, loss_fn)

    writer.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
