#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import cart_pole_pixels_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Running in ReCodEx"
)
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument(
    "--hidden_layer_size", default=128, type=int, help="Size of hidden layer."
)
parser.add_argument(
    "--hidden_blocks", default=2, type=int, help="Num of hidden blocks."
)
parser.add_argument(
    "--model_learning_rate", default=0.005, type=float, help="Model learning rate."
)
parser.add_argument(
    "--baseline_learning_rate",
    default=0.05,
    type=float,
    help="Baseline learning rate.",
)
parser.add_argument("--gamma", default=0.99, type=float, help="Gamma factor.")
parser.add_argument("--filters", default=16, type=int, help="")
parser.add_argument(
    "--block_sizes",
    nargs="+",
    type=int,
    default=[2],
    help="Individual dense block sizes",
)
parser.add_argument(
    "--dropout_rate", default=0.3, type=float, help="Dropout rate for both models."
)
parser.add_argument(
    "--regularization_rate",
    default=0.01,
    type=float,
    help="Regularization rate for both models.",
)
parser.add_argument(
    "--save",
    default=False,
    type=bool,
    help="Determines whether to save the trained model.",
)


class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model. The predict method assumes
        # the policy network is stored as `self._model`.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing the baseline (with a single output without an activation).
        # (Alternatively, this baseline computation can be grouped together
        # with the policy computation in a single `tf.keras.Model`.)
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
        self._input_shape = env.observation_space.shape
        if os.path.exists("reinforce_pixels_model.h5") and os.path.exists(
            "reinforce_pixels_baseline.h5"
        ):
            self._model = tf.keras.models.load_model("reinforce_pixels_model.h5")
            self._baseline = tf.keras.models.load_model("reinforce_pixels_baseline.h5")
        else:

            def _dense_block_part(hidden, filters, kernel, activation="relu"):
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden = tf.keras.layers.Activation(activation)(hidden)
                return tf.keras.layers.Conv2D(filters, kernel, 1, padding="same")(
                    hidden
                )

            def dense_block(hidden, filters, dense_block_size):
                for _ in range(dense_block_size):
                    hidden_part = _dense_block_part(hidden, 4 * filters, 1)
                    hidden_part = _dense_block_part(hidden_part, filters, 3)
                    hidden = tf.keras.layers.Concatenate()([hidden_part, hidden])
                return hidden

            def transition_layer(hidden):
                hidden = _dense_block_part(hidden, hidden.shape[-1] // 2, 1)
                return tf.keras.layers.MaxPool2D(2, 2, padding="same")(hidden)

            inputs = tf.keras.layers.Input(
                shape=env.observation_space.shape, dtype=tf.float32
            )
            hidden = tf.keras.layers.Conv2D(2 * args.filters, 7, 2, "same")(inputs)
            hidden = tf.keras.layers.MaxPooling2D(3, 2)(hidden)

            for dense_block_size in args.block_sizes:
                hidden_part = dense_block(hidden, args.filters, dense_block_size)
                hidden = transition_layer(hidden_part)

            hidden = tf.keras.layers.GlobalAveragePooling2D()(hidden_part)

            outs = []
            for _ in range(2):
                x = hidden
                x = tf.keras.layers.Dense(
                    args.hidden_layer_size,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(
                        l1=args.regularization_rate, l2=args.regularization_rate
                    ),
                    activation=None,
                )(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation("swish")(x)
                x = tf.keras.layers.Dropout(args.dropout_rate)(x)
                for _ in range(args.hidden_blocks):
                    y = tf.keras.layers.Dense(
                        args.hidden_layer_size,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(
                            l1=args.regularization_rate, l2=args.regularization_rate
                        )
                        if args.regularization_rate
                        else None,
                        activation=None,
                    )(x)
                    y = tf.keras.layers.BatchNormalization()(y)
                    y = tf.keras.layers.Activation("swish")(y)
                    y = tf.keras.layers.Dropout(args.dropout_rate)(y)
                    x = tf.keras.layers.Add()([x, y])
                outs.append(x)

            model_outputs = tf.keras.layers.Dense(
                env.action_space.n, activation="softmax"
            )(outs[0])
            baseline_outputs = tf.keras.layers.Dense(1, activation=None)(outs[1])

            self._model = tf.keras.Model(inputs=inputs, outputs=model_outputs)
            self._baseline = tf.keras.Model(inputs=inputs, outputs=baseline_outputs)

        self._baseline.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.baseline_learning_rate,
            ),
        )
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.model_learning_rate,
            )
        )

    # Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32, np.int64)
    @tf.function(experimental_relax_shapes=True)
    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        row_lengths: np.ndarray,
    ) -> None:
        # TODO: Perform training, using the loss from the REINFORCE with
        # baseline algorithm.
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the baseline model to predict `returns`
        # - train the policy model, using `returns - predicted_baseline` as
        #   the advantage estimate
        mask = tf.sequence_mask(row_lengths, dtype=tf.float32)
        with tf.GradientTape() as baseline_tape:
            deltas = mask * tf.reshape(
                self._baseline(
                    tf.reshape(states, [-1, *self._input_shape]), training=True
                ),
                [states.shape[0], -1],
            )
            baseline_loss = tf.keras.losses.MeanSquaredError()(returns, deltas)

        self._baseline.optimizer.minimize(
            baseline_loss, self._baseline.trainable_variables, tape=baseline_tape
        )

        with tf.GradientTape() as model_tape:
            probs = tf.expand_dims(mask, -1) * tf.reshape(
                self._model(
                    tf.reshape(states, [-1, *self._input_shape]), training=True
                ),
                [states.shape[0], -1, 2],
            )
            model_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
                actions, probs, sample_weight=(returns - deltas)
            )

        self._model.optimizer.minimize(
            model_loss, self._model.trainable_variables, tape=model_tape
        )

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)

    def save(self):
        self._model.save("reinforce_pixels_model.h5")
        self._baseline.save("reinforce_pixels_baseline.h5")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    agent = Agent(env, args)

    if args.recodex:
        # TODO: Perform evaluation of a trained model.
        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                if (
                    args.render_each
                    and env.episode > 0
                    and env.episode % args.render_each == 0
                ):
                    env.render()
                # TODO: Choose an action
                action = np.argmax(agent.predict([state]))
                state, reward, done, _ = env.step(action)

    else:
        # TODO: Perform training
        # Training
        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns, row_lengths = [], [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset(), False
                while not done:
                    if (
                        args.render_each
                        and env.episode > 0
                        and env.episode % args.render_each == 0
                    ):
                        env.render()

                    # TODO(reinforce): Choose `action` according to probabilities
                    # distribution (see `np.random.choice`), which you
                    # can compute using `agent.predict` and current `state`.
                    action = np.random.choice(
                        env.action_space.n,
                        p=np.squeeze(agent.predict([state])),
                    )

                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state

                # TODO(reinforce): Compute returns from the received rewards
                G = 0.0
                returns = []

                for reward in reversed(rewards):
                    G = reward + args.gamma * G
                    returns.append(G)

                returns.reverse()

                # TODO(reinforce): Add states, actions and returns to the training batch
                batch_states.append(states)
                batch_actions.append(actions)
                batch_returns.append(returns)
                row_lengths.append(len(returns))

            max_row_len = np.max(row_lengths)

            batch_states_np, batch_actions_np, batch_returns_np = (
                np.zeros(
                    [args.batch_size, max_row_len, *env.observation_space.shape],
                    dtype=np.float32,
                ),
                np.zeros([args.batch_size, max_row_len], dtype=np.int32),
                np.zeros([args.batch_size, max_row_len], dtype=np.int32),
            )

            for i in range(args.batch_size):
                batch_states_np[i, : row_lengths[i], ...] = batch_states[i]
                batch_actions_np[i, : row_lengths[i]] = batch_actions[i]
                batch_returns_np[i, : row_lengths[i]] = batch_returns[i]

            # TODO(reinforce): Train using the generated batch.
            agent.train(
                batch_states_np, batch_actions_np, batch_returns_np, row_lengths
            )

        if args.save:
            agent.save()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
