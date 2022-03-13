#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
import re
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument(
    "--hidden_layer", default=100, type=int, help="Size of the hidden layer."
)
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# If you add more arguments, ReCodEx will keep them with your default values.


def categorical_crossentropy_derivative(labels, probs):
    return probs - labels


def tanh_derivative(x):
    return 1 - tf.nn.tanh(x) * tf.nn.tanh(x)


class Model(tf.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        self._W1 = tf.Variable(
            tf.random.normal(
                [MNIST.W * MNIST.H * MNIST.C, args.hidden_layer],
                stddev=0.1,
                seed=args.seed,
            ),
            trainable=True,
        )
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)

        # TODO(sgd_backpropagation): Create variables:
        # - _W2, which is a trainable Variable of size [args.hidden_layer, MNIST.LABELS],
        #   initialized to `tf.random.normal` value with stddev=0.1 and seed=args.seed,
        # - _b2, which is a trainable Variable of size [MNIST.LABELS] initialized to zeros
        self._W2 = tf.Variable(
            tf.random.normal(
                [args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed
            ),
            trainable=True,
        )
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # TODO(sgd_backpropagation): Define the computation of the network. Notably:
        # - start by reshaping the inputs to shape [inputs.shape[0], -1].
        #   The -1 is a wildcard which is computed so that the number
        #   of elements before and after the reshape fits.
        # - then multiply the inputs by `self._W1` and then add `self._b1`
        # - apply `tf.nn.tanh`
        # - multiply the result by `self._W2` and then add `self._b2`
        # - finally apply `tf.nn.softmax` and return the result

        # TODO: In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tf.nn.tanh, and the input layer after reshaping.
        input_layer = tf.reshape(inputs, [inputs.shape[0], -1])
        hidden_layer = tf.nn.tanh(input_layer @ self._W1 + self._b1)
        output_layer = tf.nn.softmax(hidden_layer @ self._W2 + self._b2)
        return output_layer, hidden_layer, input_layer

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            # TODO: Contrary to sgd_backpropagation, the goal here is to compute
            # the gradient manually, without tf.GradientTape. ReCodEx checks
            # that `tf.GradientTape` is not used and if it is, your solution does
            # not pass.
            #

            y_hat, h, x = self.predict(batch["images"])
            y = tf.one_hot(batch["labels"], depth=MNIST.LABELS)
            L = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(y, y_hat))
            # TODO: Compute the input layer, hidden layer and output layer
            # of the batch images using `self.predict`.

            # TODO: Compute the gradient of the loss with respect to all
            # variables. Note that the loss is computed as in `sgd_backpropagation`:
            # - For every batch example, the loss is the categorical crossentropy of the
            #   predicted probabilities and the gold label. To compute the crossentropy, you can
            #   - either use `tf.one_hot` to obtain one-hot encoded gold labels,
            #   - or use `tf.gather` with `batch_dims=1` to "index" the predicted probabilities.
            # - Finally, compute the average across the batch examples.
            #
            # During the gradient computation, you will need to compute
            # a so-called outer product
            #   `C[a, i, j] = A[a, i] * B[a, j]`
            # which you can for example as
            #   `A[:, :, tf.newaxis] * B[:, tf.newaxis, :]`
            # or with
            #   `tf.einsum("ai,aj->aij", A, B)`
            # print("y_hat shape:", y_hat.shape)
            # print("h shape", h.shape)
            # print("x shape", x.shape)
            # print("y shape", y.shape)
            # print("L shape", L.shape)
            # print("W1 shape", self._W1.shape)
            # print("W2 shape", self._W2.shape)
            # print("b1 shape", self._b1.shape)
            # print("b2 shape", self._b2.shape)
            delta1 = y_hat - y
            # print("delta1 shape", delta1.shape)
            tanh_der = 1 - h**2
            # print("tanh derivative shape", tanh_der.shape)
            # delta2 = delta1 * tf.einsum("ij,ki->kj", self._W2, tanh_der)
            delta2 = tf.einsum("ij,kj->ik", delta1, self._W2) * tanh_der
            # print("delta2 shape", delta2.shape)

            W2_grad = tf.reduce_mean(tf.einsum("ij,ik->ikj", delta1, h), 0)
            # print("W2_grad shape", W2_grad.shape)
            b2_grad = tf.reduce_mean(delta1, 0)
            # print("b2_grad shape", b2_grad.shape)
            W1_grad = tf.reduce_mean(tf.einsum("ij,ik->ikj", delta2, x), 0)
            b1_grad = tf.reduce_mean(delta2, 0)

            # print("W1 grad:", W1_grad)
            # print("b1 grad:", b1_grad)
            # print("W2 grad:", W2_grad)
            # print("b2 grad:", b2_grad)
            # TODO(sgd_backpropagation): Perform the SGD update with learning rate `self._args.learning_rate`
            # for the variable and computed gradient. You can modify
            # variable value with `variable.assign` or in this case the more
            # efficient `variable.assign_sub`.
            self._W1 = self._W1.assign_sub(self._args.learning_rate * W1_grad)
            self._b1 = self._b1.assign_sub(self._args.learning_rate * b1_grad)
            self._W2 = self._W2.assign_sub(self._args.learning_rate * W2_grad)
            self._b2 = self._b2.assign_sub(self._args.learning_rate * b2_grad)

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            # TODO(sgd_backpropagation): Compute the probabilities of the batch images
            probabilities, _, _ = self.predict(batch["images"])

            # TODO(sgd_backpropagation): Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            correct += tf.reduce_sum(
                tf.cast(
                    tf.equal(batch["labels"], tf.argmax(probabilities, 1)), tf.float64
                )
            )

        return correct / dataset.size


def main(args: argparse.Namespace) -> float:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

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

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # TODO(sgd_backpropagation): Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(mnist.train)

        # TODO(sgd_backpropagation): Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = model.evaluate(mnist.dev)
        print(
            "Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy),
            flush=True,
        )
        with writer.as_default(step=epoch + 1):
            tf.summary.scalar("dev/accuracy", 100 * accuracy)

    # TODO(sgd_backpropagation): Evaluate the test data using `evaluate` on `mnist.test` dataset
    accuracy = model.evaluate(mnist.test)
    print(
        "Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy),
        flush=True,
    )
    with writer.as_default(step=epoch + 1):
        tf.summary.scalar("test/accuracy", 100 * accuracy)

    # Return test accuracy for ReCodEx to validate
    return accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
