#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--activation", default="none", type=str, help="Activation function.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--hidden_layers", default=1, type=int, help="Number of layers.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

ACTIVATION_FUNCS = {
    "none": None,
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid
}

def main(args: argparse.Namespace) -> float:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    # TODO: Finish the model. Namely add:
    # - a `tf.keras.layers.Flatten()` layer
    # - `args.hidden_layers` number of fully connected hidden layers
    #   `tf.keras.layers.Dense()` with  `args.hidden_layer` neurons, using activation
    #   from `args.activation`, allowing "none", "relu", "tanh", "sigmoid".
    # - finally, a final fully connected layer with
    #   `MNIST.LABELS` units and `tf.nn.softmax` activation.

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C]))
    model.add(tf.keras.layers.Flatten(name="flatten"))
    for i in range(args.hidden_layers):
        model.add(tf.keras.layers.Dense(
            args.hidden_layer,
            activation=ACTIVATION_FUNCS[args.activation],
            name="hidden_{}".format(i + 1)
        ))
    model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax, name="output_layer"))
    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.
    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    # Return test accuracy for ReCodEx to validate
    return test_logs["accuracy"]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
