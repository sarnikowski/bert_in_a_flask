import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization import FullTokenizer

from utils_bert import convert_data, create_learning_rate_scheduler
from utils_plotting import plot_confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

LOG_PREFIX = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = "/app/models/{}".format(LOG_PREFIX)

logger = tf.get_logger()

batch_size = 12
max_learn_rate = 2e-5
end_learn_rate = 2e-7
warmup_proportion = 0.1
n_epochs = 10
max_seq_len = 256
test_size = 0.1
validation_split = 0.1
random_state = None

pretrained_model_dir = "/app/model/".format(os.environ["BASE_MODEL_PREFIX"])
bert_ckpt_file = os.path.join(pretrained_model_dir, "bert_model.ckpt")
bert_config_file = os.path.join(pretrained_model_dir, "bert_config.json")

bert_vocab_file = os.path.join(pretrained_model_dir, "vocab.txt")

do_lower_case = "uncased" in pretrained_model_dir
tokenizer = FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=do_lower_case)

df = pd.read_csv("/app/data/stack-overflow-data.csv")
df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["tags"])
num_classes = len(label_encoder.classes_)
np.save(pretrained_model_dir + "label_encoder.npy", label_encoder.classes_)

y_train = label_encoder.transform(df_train["tags"])
y_test = label_encoder.transform(df_test["tags"])

tf.compat.v1.logging.info("***** Coverting training examples *****")
x_train = convert_data(df_train["post"], tokenizer=tokenizer, max_seq_len=max_seq_len)
tf.compat.v1.logging.info("***** Converting test examples *****")
x_test = convert_data(df_test["post"], tokenizer=tokenizer, max_seq_len=max_seq_len)
tf.compat.v1.logging.info("***** ...Finished converting data to BERT format *****")


def create_model(max_seq_len, summary=False):
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype="int32", name="input_ids")
    output = bert(input_ids)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="relu")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=num_classes, activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    if summary:
        model.summary()
    return model


model = create_model(max_seq_len, summary=True)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=model_dir)

learning_rate_callback = create_learning_rate_scheduler(
    max_learn_rate=max_learn_rate,
    end_learn_rate=end_learn_rate,
    warmup_proportion=warmup_proportion,
    n_epochs=n_epochs,
)
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
model.fit(
    x_train,
    y_train,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=[learning_rate_callback, early_stopping_callback, tensorboard_callback],
)
model.save_weights("{}/model.h5".format(model_dir))

model = create_model(max_seq_len)
model.load_weights("{}/model.h5".format(model_dir))

predictions = model.predict(x_test, batch_size=batch_size)
y_pred = np.argmax(predictions)
tf.compat.v1.logging.info(
    "***** BERT TEST ACCURACY: {} *****".format(accuracy_score(y_test, y_pred))
)
_, counts = np.unique(y_test, return_counts=True)

matrix = plot_confusion_matrix(
    y_test,
    y_pred,
    ["{} ({:d})".format(s, c) for s, c in zip(label_encoder.classes_, counts)],
    figsize=(16, 14),
    normalize=True,
    save_path=model_dir + "confusion_matrix.png",
)
