import os
import sys
import json
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import bert

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow import keras
from absl import logging as absl_logging

from utils_bert import convert_data, create_model, create_learning_rate_scheduler
from utils_plotting import plot_confusion_matrix

LOG_PREFIX = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_model_dir = os.path.join("/app/models", LOG_PREFIX)
os.mkdir(output_model_dir)

formatting = "%(asctime)s: %(levelname)s %(filename)s:%(lineno)s] %(message)s"
formatter = logging.Formatter(formatting)
absl_logging.get_absl_handler().setFormatter(formatter)

for h in tf.get_logger().handlers:
    h.setFormatter(formatter)

logger = tf.get_logger()
logger.setLevel(logging.INFO)

batch_size = 16
max_learn_rate = 2e-5
end_learn_rate = 2e-7
warmup_proportion = 0.2
n_epochs = 10
max_seq_len = 256
test_size = 0.1
validation_split = 0.1
random_state = None

if not os.path.isdir("/app/models/{}".format(os.environ["BASE_MODEL_PREFIX"])):
    sys.exit("Pretrained model not found.")

pretrained_model_dir = "/app/models/{}".format(os.environ["BASE_MODEL_PREFIX"])
bert_vocab_file = os.path.join(pretrained_model_dir, "vocab.txt")

do_lower_case = "uncased" in pretrained_model_dir
tokenizer = bert.bert_tokenization.FullTokenizer(
    vocab_file=bert_vocab_file, do_lower_case=do_lower_case
)

df = pd.read_csv("/app/data/stack-overflow-data.csv")
df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["tags"])
n_classes = len(label_encoder.classes_)
np.save(os.path.join(output_model_dir, "label_encoder.npy"), label_encoder.classes_)

y_train = label_encoder.transform(df_train["tags"])
y_test = label_encoder.transform(df_test["tags"])

logger.info("Converting data to Bert format...")
x_train = convert_data(df_train["post"], tokenizer=tokenizer, max_seq_len=max_seq_len)
x_test = convert_data(df_test["post"], tokenizer=tokenizer, max_seq_len=max_seq_len)
logger.info("Finished converting data to Bert format")

logger.info("x_train shape: {}".format(x_train.shape))
logger.info("y_train shape: {}".format(y_train.shape))


model = create_model(max_seq_len, pretrained_model_dir, n_classes, summary=True)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=output_model_dir)

learning_rate_callback = create_learning_rate_scheduler(
    max_learn_rate=max_learn_rate,
    end_learn_rate=end_learn_rate,
    warmup_proportion=warmup_proportion,
    n_epochs=n_epochs,
)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_model_dir, "model.h5"),
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

logger.info("Starting training of model")
model.fit(
    x_train,
    y_train,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=[learning_rate_callback, model_checkpoint, tensorboard_callback],
)
model = create_model(max_seq_len, pretrained_model_dir, n_classes, load_weights=False)
model.load_weights(os.path.join(output_model_dir, "model.h5"))

predictions = model.predict(x_test, batch_size=batch_size)
y_pred = np.argmax(predictions, axis=-1)
logger.info("Bert test accuracy: {}".format(accuracy_score(y_test, y_pred)))
_, counts = np.unique(y_test, return_counts=True)

matrix = plot_confusion_matrix(
    y_test,
    y_pred,
    ["{} ({:d})".format(s, c) for s, c in zip(label_encoder.classes_, counts)],
    figsize=(16, 14),
    normalize=True,
    save_path=os.path.join(output_model_dir, "confusion_matrix.png"),
)
model_config = {
    "max_seq_len": max_seq_len,
    "n_classes": n_classes,
    "pretrained_model_dir": pretrained_model_dir,
    "output_model_dir": output_model_dir,
}
with open(os.path.join(output_model_dir, "model_config.json"), "w") as filepath:
    json.dump(model_config, filepath)
with open("/app/models/latest_model_config.json", "w") as filepath:
    json.dump(model_config, filepath)
