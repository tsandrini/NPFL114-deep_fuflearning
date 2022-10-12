#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import random
import os
import sys
import urllib.request
from typing import Tuple, Dict, Iterator, Optional

import numpy as np
import torch

from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--examples", default=256, type=int, help="MNIST examples to use.")
parser.add_argument(
    "--iterations", default=100, type=int, help="Iterations of the power algorithm."
)
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
                torch.Tensor(self._data["images"]), torch.Tensor(self._data["labels"])
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


def main(args: argparse.Namespace) -> Tuple[float, float]:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.set_num_threads(args.threads)

    mnist = MNIST()

    data_indices = np.random.choice(mnist.train.size, size=args.examples, replace=False)
    data = torch.Tensor(mnist.train.data["images"][data_indices])
    #
    # TODO: Data has shape [args.examples, MNIST.H, MNIST.W, MNIST.C].
    # We want to reshape it to [args.examples, MNIST.H * MNIST.W * MNIST.C].
    # We can do so using `tf.reshape(data, new_shape)` with new shape
    # `[data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]`.
    data = torch.reshape(
        data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])
    )

    # TODO: Now compute mean of every feature. Use `tf.math.reduce_mean`,
    # and set `axis` to zero -- therefore, the mean will be computed
    # across the first dimension, so across examples.
    mean = torch.mean(data, dim=0)

    # TODO: Compute the covariance matrix. The covariance matrix is
    #   (data - mean)^T * (data - mean) / data.shape[0]
    # where transpose can be computed using `tf.transpose` and matrix
    # multiplication using either Python operator @ or `tf.linalg.matmul`.
    cov = (torch.transpose(data - mean, 0, 1) @ (data - mean)) / data.shape[0]

    # TODO: Compute the total variance, which is sum of the diagonal
    # of the covariance matrix. To extract the diagonal use `tf.linalg.diag_part`
    # and to sum a tensor use `tf.math.reduce_sum`.
    # total_variance = tf.linalg.trace(cov)
    total_variance = torch.sum(torch.diag(cov))

    # TODO: Now run `args.iterations` of the power iteration algorithm.
    # Start with a vector of `cov.shape[0]` ones of type tf.float32 using `tf.ones`.
    v = torch.ones((cov.shape[0]), dtype=torch.float32)
    for i in range(args.iterations):
        # TODO: In the power iteration algorithm, we compute
        # 1. v = cov * v
        #    The matrix-vector multiplication can be computed using `tf.linalg.matvec`.
        # 2. s = l2_norm(v)
        #    The l2_norm can be computed using `tf.norm`.
        # 3. v = v / s
        v = cov @ v
        s = torch.linalg.norm(v)
        v = v / s

    # The `v` is now the eigenvector of the largest eigenvalue, `s`. We now
    # compute the explained variance, which is a ration of `s` and `total_variance`.
    explained_variance = s / total_variance

    # Return the total and explained variance for ReCodEx to validate
    return total_variance, 100 * explained_variance


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance = main(args)
    print("Total variance: {:.2f}".format(total_variance))
    print("Explained variance: {:.2f}%".format(explained_variance))
