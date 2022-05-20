#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Running in ReCodEx"
)
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=1.0, type=float, help="Gamma factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Fix random seed
    np.random.seed(args.seed)

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float32)
    C = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.int32)

    for _ in range(args.episodes):
        # Perform episode, collecting states, actions and rewards
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if (
                args.render_each
                and env.episode > 0
                and env.episode % args.render_each == 0
            ):
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of `args.epsilon`, use a random action,
            # otherwise choose and action with maximum `Q[state, action]`.
            action = np.where(
                np.random.uniform() > args.epsilon,
                np.argmax(Q[state, :]),
                env.action_space.sample(),
            )

            # Perform the action.
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # TODO: Compute returns from the received rewards and update Q and C.
        G = 0.0
        for t in reversed(range(len(rewards) - 1)):
            S, A = states[t], actions[t]
            G = args.gamma * G + rewards[t + 1]
            C[S, A] += 1
            Q[S, A] += (G - Q[S, A]) / C[S, A]

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state, :])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed
    )

    main(env, args)
