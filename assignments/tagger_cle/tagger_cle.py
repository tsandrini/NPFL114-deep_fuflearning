#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
import datetime
import os
import re
from typing import Any, Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
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
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument(
    "--word_masking",
    default=0.0,
    type=float,
    help="Mask words with the given probability.",
)
# If you add more arguments, ReCodEx will keep them with your default values.

LAYERS = {
    "LSTM": tf.keras.layers.LSTM,
    "GRU": tf.keras.layers.GRU,
}


class Model(tf.keras.Model):
    # A layer setting given rate of elements to zero.
    class MaskElements(tf.keras.layers.Layer):
        def __init__(self, rate: float) -> None:
            super().__init__()
            self._rate = rate

        def get_config(self) -> Dict[str, Any]:
            return {"rate": self._rate}

        def call(self, inputs: tf.RaggedTensor, training: bool) -> tf.RaggedTensor:
            if training:
                # TODO: Generate as many random uniform numbers in range [0, 1) as there are
                # values in `tf.RaggedTensor` `inputs` using a single `tf.random.uniform` call
                # (without setting seed in any way, so with just a single parameter `shape`).
                # Then, set the values in `inputs` to zero if the corresponding generated
                # random number is less than `self._rate`.
                mask = tf.cast(
                    tf.math.greater(
                        tf.random.uniform(tf.shape(inputs.values)), self._rate
                    ),
                    tf.int64,
                )
                return inputs.with_values(inputs.values * mask)
            else:
                return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        # Implement a one-layer RNN network. The input `words` is
        # a `RaggedTensor` of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        indices = train.forms.word_mapping(words)

        # TODO: With a probability of `args.word_masking`, replace the input word by an
        # unknown word (which has index 0).
        #
        # There are two approaches you can use:
        # 1) use the above defined `MaskElements` layer, in which you need to implement
        #    one TODO note. If you do not want to implement it, you can instead
        # 2) use a `tf.keras.layers.Dropout` to achieve this, even if it is a bit
        #    hacky, because Dropout cannot process integral inputs. Start by using
        #    `tf.ones_like` to create a ragged tensor of float32 ones with the same
        #    structure as the indices of the input words, pass them through a dropout layer
        #    with `args.word_masking` rate, and finally set the input word ids to 0 where
        #    the result of dropout is zero.
        masked_indices = self.MaskElements(args.word_masking)(indices)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        we_embedding = tf.keras.layers.Embedding(
            train.forms.word_mapping.vocabulary_size(), args.we_dim
        )(masked_indices)

        # TODO: Create a vector of input words from all batches using `words.values`
        # and pass it through `tf.unique`, obtaining a list of unique words and
        # indices of the original flattened words in the unique word list.
        unique_words, unique_words_idx = tf.unique(words.values)

        # TODO: Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        letter_seqs = tf.strings.unicode_split(unique_words, "UTF-8")

        # TODO: Map the letters into ids by using `char_mapping` of `train.forms`.
        letter_ids = train.forms.char_mapping(letter_seqs)

        # TODO: Embed the input characters with dimensionality `args.cle_dim`.
        char_embedding = tf.keras.layers.Embedding(
            train.forms.char_mapping.vocabulary_size(), args.cle_dim
        )(letter_ids)

        # TODO: Pass the embedded letters through a bidirectional GRU layer
        # with dimensionality `args.cle_dim`, obtaining character-level representations
        # of the whole words, **concatenating** the outputs of the forward and backward RNNs.
        cle_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(args.cle_dim), merge_mode="concat"
        )(char_embedding)

        # TODO: Use `tf.gather` with the indices generated by `tf.unique` to transform
        # the computed character-level representations of the unique words to representations
        # of the flattened (non-unique) words.
        hey = tf.gather(cle_gru, unique_words_idx)

        # TODO: Then, convert these character-level word representations into
        # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        cle_embedding = words.with_values(hey)

        # TODO: Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        embedding = tf.keras.layers.Concatenate()([we_embedding, cle_embedding])

        # TODO(tagger_we): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.
        sequences = tf.keras.layers.Bidirectional(
            LAYERS[args.rnn_cell](
                args.rnn_cell_dim,
                return_sequences=True,
                return_state=False,
            ),
            merge_mode="sum",
        )(
            embedding
        )  # TODO embedding

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

    # Return development metrics for ReCodEx to validate
    return {
        metric: values[-1]
        for metric, values in logs.history.items()
        if metric.startswith("val_")
    }


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
