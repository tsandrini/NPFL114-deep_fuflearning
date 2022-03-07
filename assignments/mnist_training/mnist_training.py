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
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

OPTIMIZERS = {
    'SGD':tf.keras.optimizers.SGD,
    'Adam': tf.keras.optimizers.Adam
}

LR_SCHEDULES = {
    'linear': tf.optimizers.schedules.PolynomialDecay,
    'exponential': tf.optimizers.schedules.ExponentialDecay
}

def main(args: argparse.Namespace) -> float:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
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
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])

    # TODO: Use the required `args.optimizer` (either `SGD` or `Adam`).
    # For `SGD`, `args.momentum` can be specified.
    # - If `args.decay` is not specified, pass the given `args.learning_rate`
    #   directly to the optimizer as a `learning_rate` argument.
    # - If `args.decay` is set, then
    #   - for `linear`, use `tf.optimizers.schedules.PolynomialDecay` with default power=1.0
    #     using the given `args.learning_rate_final`;
    #     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay
    #   - for `exponential`, use `tf.optimizers.schedules.ExponentialDecay`
    #     and set `decay_rate` appropriately to reach `args.learning_rate_final`
    #     just after the training (and keep the default `staircase=False`).
    #     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
    #   and you should pass the created `{Polynomial,Exponential}Decay` to
    #   the optimizer using the `learning_rate` constructor argument.
    #   In both cases, `decay_steps` should be total number of optimizer
    #   updates, i.e., the total number of training batches in all epochs.
    #   The size of the training MNIST dataset is `mnist.train.size` and you
    #   can assume it is divisible by `args.batch_size`.
    #
    #   If a learning rate schedule is used, TensorBoard automatically logs the value of the
    #   learning rate after every epoch. Additionally, you can find out the current learning
    #   rate manually by using `model.optimizer.learning_rate(model.optimizer.iterations)`,
    #   so after training, this value should be `args.learning_rate_final`.


    if args.decay:
        learning_rate = LR_SCHEDULES[args.decay]
        lr_schedule_args = {
            'decay_steps': int(args.epochs * mnist.train.size / args.batch_size),
        }

        if args.decay == 'linear':
            lr_schedule_args['power'] = 1.0
            lr_schedule_args['end_learning_rate'] = args.learning_rate_final
        elif args.decay == 'exponential':
            lr_schedule_args['staircase'] = False
            lr_schedule_args['decay_rate'] = args.learning_rate_final / args.learning_rate

        learning_rate = learning_rate(args.learning_rate, **lr_schedule_args)
    else:
        learning_rate = args.learning_rate

    optimizer = OPTIMIZERS[args.optimizer]
    optimizer_args = {
        'learning_rate': learning_rate,
    }

    if args.optimizer == 'SGD' and args.momentum is not None:
        optimizer_args['momentum'] = args.momentum

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
        return lr

    optimizer = optimizer(**optimizer_args)
    lr_metric = get_lr_metric(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy"), lr_metric],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    def evaluate_test(epoch, logs):
        if epoch + 1 == args.epochs:
            test_logs = model.evaluate(
                mnist.test.data["images"], mnist.test.data["labels"], args.batch_size, return_dict=True, verbose=0,
            )
            logs.update({"val_test_" + name: value for name, value in test_logs.items()})

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_test), tb_callback],
    )
    
    # Return test accuracy for ReCodEx to validate
    return logs.history["val_test_accuracy"][-1]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
