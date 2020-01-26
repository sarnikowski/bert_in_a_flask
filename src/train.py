import os
import numpy as np

import custom_logger
import loader
import data_helper
import optimization
import plotting

from sklearn.metrics import accuracy_score
from tensorflow import keras

from model import create_model

logger = custom_logger.get_logger()

batch_size = 12
max_learn_rate = 3e-5
end_learn_rate = 3e-7
warmup_proportion = 0.2
n_epochs = 10
max_seq_len = 192
test_size = 0.1
validation_split = 0.1
random_state = None

model_name = "albert_large_v2"
pretrained_model_dir, model_dir, model_type = loader.fetch_model(model_name)

datahelper = data_helper.DataHelper(
    model_dir=pretrained_model_dir, model_type=model_type, max_seq_len=max_seq_len
)

x_train, x_test, y_train, y_test, label_encoder = datahelper.get_stackoverflow_data(
    model_dir=model_dir, test_size=test_size, random_state=random_state
)
n_classes = len(label_encoder.classes_)

model = create_model(
    model_dir=pretrained_model_dir,
    model_type=model_type,
    max_seq_len=max_seq_len,
    n_classes=n_classes,
    summary=True,
)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=model_dir)

learning_rate_callback = optimization.create_learning_rate_scheduler(
    max_learn_rate=max_learn_rate,
    end_learn_rate=end_learn_rate,
    warmup_proportion=warmup_proportion,
    n_epochs=n_epochs,
)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, "model.h5"),
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

logger.info("Starting model training.")
model.fit(
    x_train,
    y_train,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=[learning_rate_callback, model_checkpoint, tensorboard_callback],
)
model = create_model(
    model_dir=pretrained_model_dir,
    model_type=model_type,
    max_seq_len=max_seq_len,
    n_classes=n_classes,
    load_pretrained_weights=False,
)
model.load_weights(os.path.join(model_dir, "model.h5"))

predictions = model.predict(x_test, batch_size=batch_size)
y_pred = np.argmax(predictions, axis=-1)
logger.info("{} test accuracy: {}".format(model_type.upper(), accuracy_score(y_test, y_pred)))
_, counts = np.unique(y_test, return_counts=True)

matrix = plotting.plot_confusion_matrix(
    y_test,
    y_pred,
    ["{} ({:d})".format(s, c) for s, c in zip(label_encoder.classes_, counts)],
    figsize=(16, 14),
    normalize=True,
    save_path=os.path.join(model_dir, "confusion_matrix.png"),
)
model_config = {"model_dir": model_dir, "model_type": model_type, "max_seq_len": max_seq_len}
loader.export_model_config(
    model_config=model_config, pretrained_model_dir=pretrained_model_dir, datahelper=datahelper
)
