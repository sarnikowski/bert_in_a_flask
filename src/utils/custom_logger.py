import logging
import tensorflow as tf
from absl import logging as absl_logging


def get_logger():
    """Retrieves tensorflow logger and changes log formatting."""
    formatting = "%(asctime)s: %(levelname)s %(filename)s:%(lineno)s] %(message)s"
    formatter = logging.Formatter(formatting)
    absl_logging.get_absl_handler().setFormatter(formatter)

    for h in tf.get_logger().handlers:
        h.setFormatter(formatter)

    logger = tf.get_logger()
    logger.setLevel(logging.INFO)
    return logger
