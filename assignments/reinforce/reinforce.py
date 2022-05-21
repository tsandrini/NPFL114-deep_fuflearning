#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument(
    "--hidden_layer_size", default=32, type=int, help="Size of hidden layer."
)
parser.add_argument(
    "--hidden_blocks", default=5, type=int, help="Num of hidden blocks."
)
parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
parser.add_argument("--gamma", default=1.0, type=float, help="Gamma factor.")
parser.add_argument(
    "--save",
    default=False,
    type=bool,
    help="Determines whether to save the trained model.",
)


class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model. The predict method assumes
        # it is stored as `self._model`.
        #
        # Using Adam optimizer with given `args.learning_rate` is a good default.
        #
        inputs = tf.keras.layers.Input(
            shape=env.observation_space.shape, dtype=tf.float32
        )
        reg = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        x = tf.keras.layers.Dense(
            args.hidden_layer_size,
            kernel_regularizer=reg,
            activation=None,
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        for _ in range(args.hidden_blocks):
            y = tf.keras.layers.Dense(
                args.hidden_layer_size,
                kernel_regularizer=reg,
                activation=None,
            )(x)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Activation("swish")(y)
            y = tf.keras.layers.Dropout(0.3)(y)
            x = tf.keras.layers.Add()([x, y])

        outputs = tf.keras.layers.Dense(env.action_space.n, activation="softmax")(x)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        )

    # Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(
        self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray
    ) -> None:
        # TODO: Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to use the `sample_weight` argument of
        # tf.losses.Loss.__call__, but you can also construct the Loss object
        # with tf.losses.Reduction.NONE and perform the weighting manually.

        with tf.GradientTape() as tape:
            probs = tf.reshape(
                self._model(tf.reshape(states, [-1, 4]), training=True),
                [states.shape[0], -1, 2],
            )
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(
                actions, probs, sample_weight=returns
            )

        self._model.optimizer.minimize(loss, self._model.trainable_variables, tape=tape)

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the agent
    agent = Agent(env, args)

    if not os.path.exists("reinforce_agent_model_dump.h5"):
        # Training
        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
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

                    # TODO: Choose `action` according to probabilities
                    # distribution (see `np.random.choice`), which you
                    # can compute using `agent.predict` and current `state`.
                    action = np.random.choice(
                        env.action_space.n,
                        p=np.squeeze(agent.predict(np.atleast_2d(state))),
                    )

                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state

                G = 0.0
                returns = []
                for t in reversed(range(len(rewards))):
                    G += np.power(args.gamma, t) * rewards[t]
                    returns.append(G)

                returns.reverse()

                # TODO: Add states, actions and returns to the training batch
                batch_states.append(states)
                batch_actions.append(actions)
                batch_returns.append(returns)

            # TODO: Train using the generated batch.
            agent.train(batch_states, batch_actions, batch_returns)

        if args.save:
            agent._model.save("reinforce_agent_model_dump.h5")
    else:
        agent._model = tf.keras.models.load_model("reinforce_agent_model_dump.h5")

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose greedy action
            action = np.argmax(agent.predict(np.atleast_2d(state)))
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed)

    main(env, args)
