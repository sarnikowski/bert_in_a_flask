import tensorflow as tf
import numpy as np
import math


def convert_data(x, tokenizer, max_seq_len):
    """
    Creates BERT compatible data examples by adding
    ["CLS"] and ["SEP"] tokens and padding/shortening
    sequences.
    """
    x_tokens = map(tokenizer.tokenize, x)
    x_tokens = map(
        lambda tok: ["[CLS]"] + tok + ["[SEP]"]
        if len(tok) <= max_seq_len - 2
        else ["[CLS]"] + tok[: max_seq_len - 2] + ["[SEP]"],
        x_tokens,
    )
    x_token_ids = list(map(tokenizer.convert_tokens_to_ids, x_tokens))

    x_token_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), x_token_ids)
    x_token_ids = np.array(list(x_token_ids))
    return x_token_ids


def create_learning_rate_scheduler(max_learn_rate, end_learn_rate, warmup_proportion, epochs):
    def lr_scheduler(epoch):
        warmup_epoch_count = int(warmup_proportion * epochs)
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate)
                * (epoch - warmup_epoch_count + 1)
                / (epochs - warmup_epoch_count + 1)
            )
        return float(res)

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler
