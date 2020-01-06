import os
import tensorflow as tf
import numpy as np
import math

from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights


def convert_data(x, tokenizer, max_seq_len):
    """
    Creates BERT compatible data examples by adding ["CLS"] and ["SEP"] tokens and padding/shortening sequences.
    """
    input_tokens = map(tokenizer.tokenize, x)
    input_tokens = map(
        lambda tokens: ["[CLS]"] + tokens + ["[SEP]"]
        if len(tokens) <= max_seq_len - 2
        else ["[CLS]"] + tokens[: max_seq_len - 2] + ["[SEP]"],
        input_tokens,
    )
    input_ids = list(map(tokenizer.convert_tokens_to_ids, input_tokens))

    input_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), input_ids)
    input_ids = np.array(list(input_ids))
    return input_ids


def create_model(max_seq_len, pretrained_model_dir, n_classes, load_weights=True, summary=False):
    """
    Creates keras model with pretrained Bert layer.
    """
    bert_ckpt_file = os.path.join(pretrained_model_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(pretrained_model_dir, "bert_config.json")

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype="int32", name="input_ids")
    output = bert(input_ids)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=n_classes, activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
    if load_weights:
        load_stock_weights(bert, bert_ckpt_file)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    if summary:
        model.summary()
    return model


def create_learning_rate_scheduler(max_learn_rate, end_learn_rate, warmup_proportion, n_epochs):
    """
    Learning rate scheduler, that increases linearly within warmup epochs
    then exponentially decreases to end_learn_rate.
    """

    def lr_scheduler(epoch):
        warmup_epoch_count = int(warmup_proportion * n_epochs)
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate)
                * (epoch - warmup_epoch_count + 1)
                / (n_epochs - warmup_epoch_count + 1)
            )
        return float(res)

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    return learning_rate_scheduler
