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
from typing import Tuple, Dict, Iterator, Optional

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
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument(
    "--hidden_layer", default=200, type=int, help="Size of the hidden layer."
)
parser.add_argument(
    "--learning_rate", default=0.01, type=float, help="Initial learning rate."
)
parser.add_argument(
    "--learning_rate_final", default=None, type=float, help="Final learning rate."
)
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
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


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.ffw = nn.Sequential(
            nn.Linear(MNIST.H * MNIST.W * MNIST.C, args.hidden_layer), nn.ReLU()
        )
        self.logits = nn.Linear(args.hidden_layer, MNIST.LABELS)

    def forward(self, x):
        x = self.flatten(x)
        x = self.ffw(x)
        return self.logits(x)


def main(args: argparse.Namespace) -> float:
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

    model = Model(args)
    summary(model, input_size=(args.batch_size, MNIST.H, MNIST.W, MNIST.C))

    # -- Prepare data --
    def get_dataloader(name):
        dataset = getattr(mnist, name).dataset
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True if name == "train" else False,
        )

    train, dev, test = (
        get_dataloader("train"),
        get_dataloader("dev"),
        get_dataloader("test"),
    )

    # -- Loss function --
    loss_fn = nn.CrossEntropyLoss()

    # -- Optimizer ---
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum if args.momentum is not None else 0.0,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
    else:
        optimizer = None

    # -- Scheduler (opt) ---
    decay_steps = int(args.epochs * mnist.train.size / args.batch_size)
    if args.decay == "linear":
        coeff = (args.learning_rate_final / args.learning_rate) ** (1 / decay_steps)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=decay_steps, factor=coeff
        )
    elif args.decay == "exponential":
        gamma = args.learning_rate_final / args.learning_rate
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = None

    # TRAIN LOOPTY-LOOP
    # -----------------
    def reset_metrics(metrics):
        for metric in metrics.values():
            metric.reset()

    def get_log(metrics, X=None, y=None, y_hat=None, all_batches=False):
        if all_batches:
            return {key: val.compute().item() for key, val in metrics.items()}
        else:
            y_ = F.one_hot(y, MNIST.LABELS)
            proba = F.softmax(y_hat)
            y_hat_ = y_hat.argmax(1)
            return {
                "acc": metrics["acc"](y_hat, y).item(),
                "kl_div": metrics["kl_div"](proba, y_).item(),
                "auc": metrics["auc"](y_hat_, y).item(),
                "auroc": metrics["auroc"](proba, y_).item(),
                "avgprec": metrics["avgprec"](proba, y_).item(),
                "f1": metrics["f1"](y_hat, y_).item(),
            }

    metrics = {
        "acc": torchmetrics.Accuracy(),
        "kl_div": torchmetrics.KLDivergence(log_prob=False),
        "auc": torchmetrics.AUC(reorder=True),
        "auroc": torchmetrics.AUROC(num_classes=MNIST.LABELS),
        "avgprec": torchmetrics.AveragePrecision(num_classes=MNIST.LABELS),
        "f1": torchmetrics.F1Score(num_classes=MNIST.LABELS),
    }

    print("\nTraining\n---------\n")
    for epoch in range(args.epochs):

        model.train()
        with tqdm(train, unit="batch", colour="green") as pbar:
            for step, (X, y) in enumerate(pbar):

                pbar.set_description(
                    f"Epoch {epoch} (lr: {(scheduler.get_last_lr()[0]):.4f})"
                    if scheduler is not None
                    else f"Epoch {epoch}"
                )

                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_dict = get_log(metrics, X, y, pred)
                log_dict["loss"] = loss.item()
                pbar.set_postfix(**log_dict)

                if writer is not None:
                    for key, value in log_dict.items():
                        writer.add_scalar(key, value, int(epoch * len(pbar) + step))

        if scheduler is not None:
            scheduler.step()

        log_dict = get_log(metrics, all_batches=True)
        log_dict["loss"] = loss.item()
        pbar.set_postfix(**log_dict)
        reset_metrics(metrics)

        # Eval
        model.eval()
        with tqdm(dev, unit="batch", colour="blue") as pbar:
            for step, (X, y) in enumerate(pbar):

                pbar.set_description(f"Epoch {epoch} - eval")

                pred = model(X)
                loss = loss_fn(pred, y)

                log_dict = get_log(metrics, X, y, pred)
                log_dict["loss"] = loss.item()
                pbar.set_postfix(**log_dict)

                if writer is not None:
                    for key, value in log_dict.items():
                        writer.add_scalar(
                            "val_" + key, value, int(epoch * len(pbar) + step)
                        )

        log_dict = get_log(metrics, all_batches=True)
        log_dict["loss"] = loss.item()
        pbar.set_postfix(**log_dict)
        reset_metrics(metrics)

    return 0.0


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
