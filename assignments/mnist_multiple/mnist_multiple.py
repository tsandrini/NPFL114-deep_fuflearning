#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
import re
from typing import Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# If you add more arguments, ReCodEx will keep them with your default values.


# The neural network model
class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # TODO: The model starts by passing each input image through the same
        # subnetwork (with shared weights), which should perform
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature representation FI of each image.
        h1 = tf.keras.layers.Conv2D(
            10, (3, 3), strides=2, padding="valid", activation="relu"
        )
        h2 = tf.keras.layers.Conv2D(
            20, (3, 3), strides=2, padding="valid", activation="relu"
        )
        h3 = tf.keras.layers.Flatten()
        h4 = tf.keras.layers.Dense(200, activation="relu")

        def _build_model(image):
            x = h1(image)
            x = h2(x)
            x = h3(x)
            return h4(x)

        models = [_build_model(image) for image in images]

        # direct_prediction
        x = tf.keras.layers.Concatenate()(models)
        x = tf.keras.layers.Dense(200, activation="relu")(x)
        direct_pred = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        # digit_1 & digit_2
        digit_layer = tf.keras.layers.Dense(10, activation="softmax")
        digit_predictions = [digit_layer(model) for model in models]

        # indirect_prediction
        # indirect_pred = tf.keras.layers.Dense(1, activation="sigmoid")(
        #     digit_predictions
        # )
        indirect_pred = tf.cast(
            tf.math.greater(
                tf.math.argmax(digit_predictions[0], axis=1),
                tf.math.argmax(digit_predictions[1], axis=1),
            ),
            tf.int32,
        )

        # TODO: Using the computed representations, the model should produce four outputs:
        # - first, compute _direct prediction_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations FI,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output using a dense layer with `tf.nn.sigmoid` activation
        # - then, classify the computed representation FI of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation FI of the second image using
        #   the same connected layer (with shared weights) into 10 classes;
        # - finally, compute _indirect prediction_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        outputs = {
            "direct_prediction": direct_pred,
            "digit_1": digit_predictions[0],
            "digit_2": digit_predictions[1],
            "indirect_prediction": indirect_pred,
        }

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # TODO: Train the model by computing appropriate losses of
        # "direct_prediction", "digit_1", "digit_2". Regarding metrics, compute
        # the accuracy of both the direct and indirect predictions; name both
        # metrics "accuracy" (i.e., pass "accuracy" as the first argument of
        # the metric object).

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={
                "direct_prediction": tf.keras.losses.BinaryCrossentropy(),
                "digit_1": tf.keras.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_prediction": [tf.keras.metrics.BinaryAccuracy("accuracy")],
                "indirect_prediction": [tf.keras.metrics.BinaryAccuracy("accuracy")],
            },
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(
        self,
        mnist_dataset: MNIST.Dataset,
        args: argparse.Namespace,
        training: bool = False,
    ) -> tf.data.Dataset:
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices(
            (mnist_dataset.data["images"], mnist_dataset.data["labels"])
        )

        # TODO: If `training`, shuffle the data with `buffer_size=10000` and `seed=args.seed`
        if training:
            dataset = dataset.shuffle(10000, seed=args.seed)

        # TODO: Combine pairs of examples by creating batches of size 2
        dataset = dataset.batch(2)

        # TODO: Map pairs of images to elements suitable for our model. Notably,
        # the elements should be pairs `(input, output)`, with
        # - `input` being a pair of images,
        # - `output` being a dictionary with keys "digit_1", "digit_2", "direct_prediction",
        #   and "indirect_prediction".
        def create_element(images, labels):
            return (
                (images[0], images[1]),
                {
                    "digit_1": labels[0],
                    "digit_2": labels[1],
                    "direct_prediction": tf.cast(
                        tf.math.greater(labels[0], labels[1]), tf.int32
                    ),
                    "indirect_prediction": tf.cast(
                        tf.math.greater(labels[0], labels[1]), tf.int32
                    ),
                },
            )

        dataset = dataset.map(create_element)

        # TODO: Create batches of size `args.batch_size`
        dataset = dataset.batch(args.batch_size)

        return dataset


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

    # Create the model
    model = Model(args)

    # Construct suitable datasets from the MNIST data.
    train = model.create_dataset(mnist.train, args, training=True)
    dev = model.create_dataset(mnist.dev, args)

    # Train
    logs = model.fit(
        train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback]
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
