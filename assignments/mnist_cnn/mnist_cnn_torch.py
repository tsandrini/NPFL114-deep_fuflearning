#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
import sys
import re
import random
from types import prepare_class

import urllib.request
from typing import Tuple, Dict, Iterator, Optional, List, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary
from tqdm import tqdm
import torchmetrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
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


class ModelContainer:
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        metrics: Dict[str, torchmetrics.Metric] = None,
        update_metrics_fn: Callable[
            [Dict[str, torchmetrics.Metric], torch.Tensor, torch.Tensor, torch.Tensor],
            Dict[str, Union[int, float, complex]],
        ] = None,
        writer: SummaryWriter = None,
    ) -> None:
        if metrics is not None and update_metrics_fn is None:
            raise AttributeError(
                "Cannot pass metrics without their appropriate update function."
            )

        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.update_metrics_fn = update_metrics_fn
        self.metrics = metrics
        self.writer = writer

    def fit(
        self, data, validation_data=None, epochs: int = None, verbose: bool = True
    ) -> Dict[str, Union[int, float, complex]]:
        data = self.to_dataloader(data)
        if validation_data is not None:
            validation_data = self.to_dataloader(validation_data)

        epochs = epochs if epochs else 1
        for epoch in range(epochs):
            self.model.train()

            pbar = tqdm(data, unit="batch", colour="green") if verbose else data
            for step, (X, y) in enumerate(pbar):

                if verbose:
                    pbar.set_description(
                        f"Epoch {epoch} (lr: {(self.scheduler.get_last_lr()[0]):.4f})"
                        if self.scheduler is not None
                        else f"Epoch {epoch}"
                    )

                pred = self.model(X)
                if self.loss_fn:
                    loss = self.loss_fn(pred, y)

                if self.optimizer:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                log_dict = (
                    self.update_metrics_fn(self.metrics, X, y, pred)
                    if self.metrics is not None
                    else {}
                )
                if self.loss_fn:
                    log_dict["loss"] = loss.item()

                if verbose:
                    pbar.set_postfix(**log_dict)

                if self.writer:
                    for key, value in log_dict.items():
                        self.writer.add_scalar(
                            key, value, int(epoch * len(pbar) + step)
                        )

            if self.scheduler:
                self.scheduler.step()

            log_dict = self.compute_metrics() if self.metrics is not None else {}
            if self.loss_fn:
                log_dict["loss"] = loss.item()
            if verbose:
                pbar.set_postfix(**log_dict)
            if self.metrics is not None:
                self.reset_metrics()

            if validation_data is not None:
                self.evaluate(validation_data, step=epoch)

        return log_dict

    def predict(self, data) -> torch.Tensor:
        self.model.eval()
        return self.model(data)

    def evaluate(
        self, data, step: int = 0, verbose: bool = True
    ) -> Dict[str, Union[int, float, complex]]:
        self.model.eval()
        data = self.to_dataloader(data)
        pbar = tqdm(data, unit="batch", colour="blue") if verbose else data
        for step, (X, y) in enumerate(pbar):

            if verbose:
                pbar.set_description(f"Eval: ")

            pred = self.model(X)
            if self.loss_fn:
                loss = self.loss_fn(pred, y)

            log_dict = (
                self.update_metrics_fn(self.metrics, X, y, pred)
                if self.metrics is not None
                else {}
            )
            if self.loss_fn:
                log_dict["loss"] = loss.item()

            if verbose:
                pbar.set_postfix(**log_dict)

            if self.writer:
                for key, value in log_dict.items():
                    self.writer.add_scalar(
                        "val_" + key, value, int(step * len(pbar) + step)
                    )

        log_dict = self.compute_metrics() if self.metrics is not None else {}
        if self.loss_fn:
            log_dict["loss"] = loss.item()

        if verbose:
            pbar.set_postfix(**log_dict)
        if self.metrics is not None:
            self.reset_metrics()

        return log_dict

    def to_dataloader(
        self, x, batch_size: int = None, shuffle: bool = None
    ) -> DataLoader:
        if isinstance(x, DataLoader):
            return x
        elif isinstance(x, TensorDataset):
            return DataLoader(
                x,
                batch_size=batch_size if batch_size is not None else 1,
                shuffle=shuffle if shuffle is not None else False,
            )
        elif isinstance(x, tuple) or isinstance(x, list):
            x, y = x
            return self.to_dataloader(TensorDataset(torch.Tensor(x), torch.Tensor(y)))
        else:
            raise NotImplementedError()

    def reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def compute_metrics(self) -> Dict[str, Union[int, float, complex]]:
        return {key: val.compute().item() for key, val in self.metrics.items()}

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


class Model(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def main(args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

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

    # -- Prepare data --
    def get_dataloader(name):
        dataset = getattr(mnist, name).dataset
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True if name == "train" else False,
        )

    train, dev = (
        get_dataloader("train"),
        get_dataloader("dev"),
    )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
