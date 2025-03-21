#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
from platform import architecture
import re
from typing import Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

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


def _parse_layer_name_and_spec(layer_string):
    if "-" in layer_string:
        layer_name, layer_spec = layer_string.split("-", 1)
    else:
        layer_name, layer_spec = layer_string, None
    return layer_name, layer_spec


def _get_layer_dict(layer):
    layer_name, layer_spec = _parse_layer_name_and_spec(layer)
    layer_dict = {"type": layer_name, "orig_spec": layer_spec}

    if layer_name == "C":

        def _build_conv2d(layer_dict, layer_input):
            return tf.keras.layers.Conv2D(*layer_dict["args"], **layer_dict["kwargs"])(
                layer_input
            )

        filters, kernel_size, strides, padding = layer_spec.split("-")

        layer_dict["build_fn"] = _build_conv2d
        layer_dict["args"] = [int(filters), int(kernel_size)]
        layer_dict["kwargs"] = {
            "strides": int(strides),
            "padding": padding,
            "activation": "relu",
        }

    elif layer_name == "CB":

        def _build_conv2d_batchnorm(layer_dict, layer_input):
            h1 = tf.keras.layers.Conv2D(*layer_dict["args"], **layer_dict["kwargs"])(
                layer_input
            )
            h2 = tf.keras.layers.BatchNormalization()(h1)
            return tf.keras.layers.Activation("relu")(h2)

        filters, kernel_size, strides, padding = layer_spec.split("-")

        layer_dict["build_fn"] = _build_conv2d_batchnorm
        layer_dict["args"] = [int(filters), int(kernel_size)]
        layer_dict["kwargs"] = {
            "strides": int(strides),
            "padding": padding,
            "activation": None,
            "use_bias": False,
        }

    elif layer_name == "M":  # Max pooling 2D

        def _build_max_pooling(layer_dict, layer_input):
            return tf.keras.layers.MaxPooling2D(
                *layer_dict["args"], **layer_dict["kwargs"]
            )(layer_input)

        pool_size, strides = layer_spec.split("-")

        layer_dict["build_fn"] = _build_max_pooling
        layer_dict["args"] = []
        layer_dict["kwargs"] = {"pool_size": int(pool_size), "strides": int(strides)}

    elif layer_name == "R":

        def _build_residual_block(layer_dict, layer_input):
            x = layer_input
            for sublayer_dict in layer_dict["layers"]:
                x = sublayer_dict["build_fn"](sublayer_dict, x)

            return tf.keras.layers.Add()([layer_input, x])

        layer_dict["build_fn"] = _build_residual_block
        layer_dict["args"] = []
        layer_dict["kwargs"] = {}
        layer_dict["layers"] = [
            _get_layer_dict(spec) for spec in layer_spec[1:-1].split(",")
        ]

    elif layer_name == "F":  # Flatten layer

        def _build_flatten(layer_dict, layer_input):
            return tf.keras.layers.Flatten(*layer_dict["args"], **layer_dict["kwargs"])(
                layer_input
            )

        layer_dict["build_fn"] = _build_flatten
        layer_dict["args"] = []
        layer_dict["kwargs"] = {}

    elif layer_name == "H":  # Dense layer

        def _build_dense(layer_dict, layer_input):
            return tf.keras.layers.Dense(*layer_dict["args"], **layer_dict["kwargs"])(
                layer_input
            )

        layer_dict["build_fn"] = _build_dense
        layer_dict["args"] = [int(layer_spec)]
        layer_dict["kwargs"] = {"activation": "relu"}

    elif layer_name == "D":

        def _build_dropout(layer_dict, layer_input):
            return tf.keras.layers.Dropout(*layer_dict["args"], **layer_dict["kwargs"])(
                layer_input
            )

        layer_dict["build_fn"] = _build_dropout
        layer_dict["args"] = [float(layer_spec)]
        layer_dict["kwargs"] = {}

    return layer_dict


def _build_nn_spec(layers_str: str):
    arch_raw = list()
    arch_string = layers_str
    while True:
        if "," not in arch_string:
            arch_raw.append(arch_string)
            break

        left, right = arch_string.split(",", 1)
        if "[" in left:
            left, right = arch_string.split("],", 1)
            arch_raw.append(left + "]")
        else:
            arch_raw.append(left)

        arch_string = right

    architecture = list()
    for layer in arch_raw:
        architecture.append(_get_layer_dict(layer))
    return architecture


# The neural network model
class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        layer_dicts = _build_nn_spec(args.cnn)
        hidden = inputs
        for layer_dict in layer_dicts:
            hidden = layer_dict["build_fn"](layer_dict, hidden)

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in the variable `hidden`.
        # hidden = ...

        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
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

    # Load the data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"],
        mnist.train.data["labels"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate
    return {
        metric: values[-1]
        for metric, values in logs.history.items()
        if metric.startswith("val_")
    }


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
