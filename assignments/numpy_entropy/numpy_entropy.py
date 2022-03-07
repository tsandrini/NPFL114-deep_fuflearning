#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
from typing import Tuple
from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_orig = defaultdict(lambda: 0)
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_orig[line] += 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_dist = np.array(list(data_orig.values()), dtype=np.int32)
    data_dist = data_dist / np.sum(data_dist)

    # TODO: Load model distribution, each line `string \t probability`.
    model_orig = defaultdict(lambda: .0)
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            key, val = line.split("\t")
            model_orig[key] = float(val)

    # TODO: Create a NumPy array containing the model distribution.

    is_zero = any([model_orig[x] == .0 for x in data_orig.keys()])

    if not is_zero:
        pasta_pesto_peperoni = data_orig.keys()
        model_orig = {key: model_orig[key] for key in pasta_pesto_peperoni}

    model_dist = np.array(list(model_orig.values()), dtype=np.float32)

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - np.sum(data_dist * np.log(data_dist))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    crossentropy = np.inf if is_zero else (-np.sum(data_dist * np.log(model_dist)))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = np.inf if is_zero else (np.sum(data_dist * np.log(data_dist / model_dist)))

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
