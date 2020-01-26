import math
import tensorflow as tf


def create_learning_rate_scheduler(max_learn_rate, end_learn_rate, warmup_proportion, n_epochs):
    """Learning rate scheduler, that increases linearly within warmup epochs
    then exponentially decreases to end_learn_rate.

    Args:
        max_learn_rate: Float. Maximum learning rate.
        end_learn_rate: Float. Scheduler converges to this value.
        warmup_proportion: Float. How many epochs to increase linearly, before decaying.
        n_epochs: Float. Maximum number of epochs training will run.

    Returns:
        Keras learning rate scheduler
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
