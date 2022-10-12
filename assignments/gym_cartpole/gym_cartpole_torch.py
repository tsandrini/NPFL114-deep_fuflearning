#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
import sys
import re
import random

import urllib.request
from typing import Tuple, Dict, Iterator, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary
from tqdm import tqdm

# from torchviz import make_dot


def evaluate_model(
    model,
    seed: int = 42,
    episodes: int = 100,
    render: bool = False,
    report_per_episode: bool = False,
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gym

    # Create the environment
    env = gym.make("CartPole-v1")
    env.seed(seed)

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset(), 0, False
        while not done:
            if render:
                env.render()

            with torch.no_grad():
                prediction = (
                    F.softmax(model(torch.Tensor(observation[np.newaxis, ...])))[0]
                ).numpy()

            if len(prediction) == 1:
                action = 1 if prediction[0] > 0.5 else 0
            elif len(prediction) == 2:
                action = np.argmax(prediction)
            else:
                raise ValueError(
                    "Unknown model output shape, only 1 or 2 outputs are supported"
                )

            observation, reward, done, info = env.step(action)
            score += reward

        total_score += score
        if report_per_episode:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--evaluate", default=False, action="store_true", help="Evaluate the given model"
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument(
    "--render", default=False, action="store_true", help="Render during evaluation"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument(
    "--hidden_layer", default=128, type=int, help="Size of the hidden layer."
)
parser.add_argument(
    "--dropout_rate", default=0.3, type=int, help="Size of the hidden layer."
)
parser.add_argument(
    "--hidden_layers", default=128, type=int, help="Size of the hidden layer."
)
parser.add_argument(
    "--model",
    default="gym_cartpole_model_torch.pth",
    type=str,
    help="Output model path.",
)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.transformation_layer = nn.Linear(4, args.hidden_layer)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(args.hidden_layer, args.hidden_layer),
                    nn.BatchNorm1d(num_features=args.hidden_layer),
                    nn.SiLU(),
                    nn.Dropout(args.dropout_rate),
                )
                for _ in range(args.hidden_layers)
            ]
        )

        self.logits = nn.Linear(args.hidden_layer, 2)

    def forward(self, x):
        x = self.transformation_layer(x)
        for block in self.blocks:
            x = x + block(x)
        return self.logits(x)


def main(args: argparse.Namespace):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    torch.set_num_threads(args.threads)

    if not args.evaluate:
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
        writer = SummaryWriter(args.logdir)

        model = Model(args)
        summary(model, input_size=(args.batch_size, 4))
        # make_dot(
        #     model(torch.randn(1, 4)).mean(), params=dict(model.named_parameters())
        # ).render("rnn_torchviz", format="png")

        data = np.loadtxt("gym_cartpole_data.txt")
        observations, labels = data[:, :-1], data[:, -1].astype(np.int32)

        train_dataset = TensorDataset(
            torch.Tensor(observations), torch.Tensor(labels).type(torch.LongTensor)
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * args.batch_size
        )

        # TRAIN LOOPTY-LOOP
        # -----------------
        print("\nTraining\n---------\n")
        model.train()
        for epoch in range(args.epochs):
            with tqdm(train_dataloader, unit="batch", colour="green") as pbar:
                for step, (X, y) in enumerate(pbar):

                    pbar.set_description(
                        f"Epoch {epoch} (lr: {(scheduler.get_last_lr()[0]):.4f})"
                    )

                    pred = model(X)
                    loss = loss_fn(pred, y)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    accuracy = (pred.argmax(1) == y).type(
                        torch.float
                    ).sum().item() / len(X)
                    pbar.set_postfix(loss=loss.item(), accuracy=100.0 * accuracy)

                    if writer is not None:
                        writer.add_scalar("loss", loss, epoch * (step / len(pbar)))

            scheduler.step()

        torch.save(model, args.model)
    else:
        model = torch.load(args.model)
        model.eval()

        if args.recodex:
            return model
        else:
            score = evaluate_model(
                model, seed=args.seed, render=args.render, report_per_episode=True
            )
            print("The average score was {}.".format(score))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
