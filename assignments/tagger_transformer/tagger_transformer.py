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

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument(
    "--max_sentences",
    default=None,
    type=int,
    help="Maximum number of sentences to load.",
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument(
    "--transformer_dropout", default=0.0, type=float, help="Transformer dropout."
)
parser.add_argument(
    "--transformer_expansion",
    default=4,
    type=float,
    help="Transformer FFN expansion factor.",
)
parser.add_argument(
    "--transformer_heads", default=4, type=int, help="Transformer heads."
)
parser.add_argument(
    "--transformer_layers", default=2, type=int, help="Transformer layers."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.dense_layer = tf.keras.layers.Dense(
                self.dim * self.expansion, activation="relu"
            )
            self.dense_downsample = tf.keras.layers.Dense(self.dim, activation=None)

        def get_config(self):
            return {"dim": self.dim, "expansion": self.expansion}

        def call(self, inputs):
            # TODO: Execute the FFN Transformer layer.
            hidden = self.dense_layer(inputs)
            hidden = self.dense_downsample(hidden)
            return hidden

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; for other arguments, keep the default values
            # (which mean trainable float32 matrices initialized with `"glorot_uniform"`).
            self.W_Q = self.add_weight(
                shape=[self.dim, self.dim],
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
            )
            self.W_K = self.add_weight(
                shape=[self.dim, self.dim],
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
            )
            self.W_V = self.add_weight(
                shape=[self.dim, self.dim],
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
            )
            self.W_O = self.add_weight(
                shape=[self.dim, self.dim],
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
            )

        def get_config(self):
            return {"dim": self.dim, "heads": self.heads}

        def call(self, inputs, mask):
            def process_weights(matrix_weight):
                return tf.transpose(
                    tf.reshape(
                        inputs @ matrix_weight,
                        tf.concat(
                            [
                                tf.shape(inputs)[:2],
                                [self.heads, self.dim // self.heads],
                            ],
                            axis=0,
                        ),
                    ),
                    perm=[0, 2, 1, 3],
                )

            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to [batch_size, max_sentence_len, heads, dim // heads],
            # - transpose via `tf.transpose` to [batch_size, heads, max_sentence_len, dim // heads].
            Q, K, V = (
                process_weights(self.W_Q),
                process_weights(self.W_K),
                process_weights(self.W_V),
            )

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            self_attention = (tf.matmul(Q, K, transpose_b=True)) / tf.math.sqrt(
                tf.cast(self.dim // self.heads, tf.float32)
            )

            # TODO: Apply the softmax, but including a suitable mask, which ignores all padding words.
            # The original `mask` is a bool matrix of shape [batch_size, max_sentence_len]
            # indicating which words are valid (True) or padding (False).
            # - You can perform the masking manually, by setting the attention weights
            #   of padding words to -1e9.
            # - Alternatively, you can use the fact that tf.keras.layers.Softmax accepts a named
            #   boolean argument `mask` indicating the valid (True) or padding (False) elements.
            self_attention = tf.keras.layers.Softmax()(
                self_attention,
                mask=mask[:, None, None, :],
            )

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - transpose the result to [batch_size, max_sentence_len, heads, dim // heads],
            # - reshape to [batch_size, max_sentence_len, dim],
            # - multiply the result by the W_O matrix.

            return (
                tf.reshape(
                    tf.transpose(
                        self_attention @ V,
                        perm=[0, 2, 1, 3],
                    ),
                    tf.concat([tf.shape(inputs)[:2], [self.dim]], axis=0),
                )
                @ self.W_O
            )

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            # Make sure `dim` is even.
            assert dim % 2 == 0

            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = (
                layers,
                dim,
                expansion,
                heads,
                dropout,
            )
            self.layer_norm1 = list()
            self.self_attn = list()
            self.dropout1 = list()
            self.layer_norm2 = list()
            self.ffn = list()
            self.dropout2 = list()
            # TODO: Create the required number of transformer layers, each consisting of
            # - a layer normalization and a self-attention layer followed by a dropout layer,
            # - a layer normalization and a FFN layer followed by a dropout layer.
            for _ in range(self.layers):
                self.layer_norm1.append(tf.keras.layers.LayerNormalization())
                self.self_attn.append(Model.SelfAttention(self.dim, self.heads))
                self.dropout1.append(tf.keras.layers.Dropout(self.dropout))
                self.layer_norm2.append(tf.keras.layers.LayerNormalization())
                self.ffn.append(Model.FFN(self.dim, self.expansion))
                self.dropout2.append(tf.keras.layers.Dropout(self.dropout))

        def get_config(self):
            return {
                name: getattr(self, name)
                for name in ["layers", "dim", "expansion", "heads", "dropout"]
            }

        def call(self, inputs, mask):
            # TODO: Start by computing the sinusoidal positional embeddings.
            # They have a shape `[max_sentence_len, dim]`, where `dim` is even and
            # - for `0 <= i < dim / 2`, the value on index `[pos, i]` should be
            #     `sin(pos / 10000 ** (2 * i / dim))`
            # - the value on index `[pos, i]` for `i >= dim / 2` should be
            #     `cos(pos / 10000 ** (2 * (i - dim/2) / dim))`
            # - the `0 <= pos < max_sentence_len` is a sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            max_sentence_len = tf.shape(inputs)[1]
            pos_sin = tf.math.sin(
                (lambda pos, i, dim: pos / tf.math.pow(10000.0, (2 * i) / dim))(
                    tf.range(max_sentence_len, dtype=tf.float32)[:, tf.newaxis],
                    tf.range(self.dim / 2, dtype=tf.float32)[tf.newaxis, :],
                    self.dim,
                )
            )
            pos_cos = tf.math.cos(
                (
                    lambda pos, i, dim: pos
                    / tf.math.pow(10000.0, (2 * ((i - (dim / 2))) / dim))
                )(
                    tf.range(max_sentence_len, dtype=tf.float32)[:, tf.newaxis],
                    tf.range(self.dim / 2, self.dim, dtype=tf.float32)[tf.newaxis, :],
                    self.dim,
                )
            )

            pos_embeddings = tf.concat([pos_sin, pos_cos], axis=1)[tf.newaxis, ...]

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layer, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            x = inputs + pos_embeddings[:, :max_sentence_len, :]
            for i in range(self.layers):
                layer_norm1 = self.layer_norm1[i](x)
                attn = self.self_attn[i](layer_norm1, mask=mask)
                dropout1 = self.dropout1[i](attn)
                x += dropout1

                layer_norm2 = self.layer_norm2[i](x)
                ffn = self.ffn[i](layer_norm2)
                dropout2 = self.dropout2[i](ffn)
                x += dropout2
            return x

    def __init__(self, args, train):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        indices = train.forms.word_mapping(words)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        embedding = tf.keras.layers.Embedding(
            train.forms.word_mapping.vocabulary_size(), args.we_dim
        )(indices)

        # TODO: Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        sequences = tf.RaggedTensor.from_tensor(
            self.Transformer(
                layers=args.transformer_layers,
                dim=args.we_dim,
                expansion=args.transformer_expansion,
                heads=args.transformer_heads,
                dropout=args.transformer_dropout,
            )(embedding.to_tensor(), mask=tf.sequence_mask(embedding.row_lengths())),
            lengths=embedding.row_lengths(),
        )

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        predictions = tf.keras.layers.Dense(
            train.tags.word_mapping.vocabulary_size(), activation="softmax"
        )(sequences)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=lambda yt, yp: tf.losses.SparseCategoricalCrossentropy()(
                yt.values, yp.values
            ),
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # TODO(tagger_we): Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integral tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.
    def extract_tagging_data(example):
        return (example["forms"], morpho.train.tags.word_mapping(example["tags"]))

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = (
            dataset.shuffle(len(dataset), seed=args.seed)
            if name == "train"
            else dataset
        )
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(args.batch_size)
        )
        return dataset

    train, dev = create_dataset("train"), create_dataset("dev")

    logs = model.fit(
        train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback]
    )

    # Return development and training losses for ReCodEx to validate
    return {
        metric: values[-1]
        for metric, values in logs.history.items()
        if "loss" in metric
    }


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
