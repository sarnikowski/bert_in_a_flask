import yaml
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

from bert import run_classifier
from bert import tokenization

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from datetime import datetime
from pathlib import Path, PurePath

from utils_plotting import plot_confusion_matrix
from utils_bert import create_examples, get_estimator, create_serving_input_receiver_fn
from utils_bert import convert_examples_to_features
from best_checkpoints_exporter import BestCheckpointsExporter

###################
# GLOBAL SETTINGS #
###################
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

log_dir = "/app/logs/{}/".format(datetime.now().strftime("%Y%m%dT%H%M%S"))
Path(log_dir).mkdir(parents=True)

logger = tf.get_logger()
#################
# BERT SETTINGS #
#################
train_batch_size = 12
learning_rate = 2e-5
num_train_epochs = 0.2
max_seq_length = 192
warmup_proportion = 0.1
test_size = 0.1
random_state = None

# Model files/paths
pretrained_model_dir = "/app/models/pretrained/"
tmp_checkpoint_dir = "/app/models/train/"
best_checkpoint_dir = "/app/models/best_checkpoint/"
trained_model_dir = "/app/models/trained_model/"

bert_vocab = pretrained_model_dir + "vocab.txt"
bert_config_path = pretrained_model_dir + "bert_config.json"
do_lower_case = "uncased" in pretrained_model_dir
init_checkpoint = pretrained_model_dir + "bert_model.ckpt"

Path(best_checkpoint_dir).mkdir(exist_ok=True)

save_checkpoints_steps = 2000
save_summary_steps = 250
keep_checkpoint_max = 3
################
# DATA LOADING #
################
df = pd.read_csv("/app/data/stack-overflow-data.csv")
df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
df_train, df_eval = train_test_split(df_train, test_size=test_size, random_state=random_state)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["tags"])
np.save(pretrained_model_dir + "label_encoder.npy", label_encoder.classes_)

y_train = label_encoder.transform(df_train["tags"])
y_eval = label_encoder.transform(df_eval["tags"])
y_test = label_encoder.transform(df_test["tags"])

num_labels = len(label_encoder.classes_)
label_list = [str(num) for num in range(num_labels)]

tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab, do_lower_case=do_lower_case)

tf.compat.v1.logging.info("***** Converting data to BERT format... *****")

tf.compat.v1.logging.info("***** Coverting training examples *****")
train_examples = create_examples(df_train["post"], y_train)
train_features = convert_examples_to_features(
    train_examples, label_list, max_seq_length, tokenizer
)

tf.compat.v1.logging.info("***** Converting evaluation examples *****")
eval_examples = create_examples(df_eval["post"], y_eval)
eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)

tf.compat.v1.logging.info("***** Converting test examples *****")
test_examples = create_examples(df_test["post"], y_test)
test_features = convert_examples_to_features(test_examples, label_list, max_seq_length, tokenizer)

tf.compat.v1.logging.info("***** ...Finished converting data to BERT format *****")
#############
# INIT BERT #
#############
num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
num_warmup_steps = int(num_train_steps * warmup_proportion)

model_config = {
    "model_dir": tmp_checkpoint_dir,
    "save_summary_steps": save_summary_steps,
    "save_checkpoints_steps": save_checkpoints_steps,
    "keep_checkpoint_max": keep_checkpoint_max,
    "bert_config_path": bert_config_path,
    "num_labels": num_labels,
    "init_checkpoint": init_checkpoint,
    "learning_rate": learning_rate,
    "num_train_steps": num_train_steps,
    "num_warmup_steps": num_warmup_steps,
    "train_batch_size": train_batch_size,
    "max_seq_length": max_seq_length,
    "do_lower_case": do_lower_case,
    "vocab_file": bert_vocab,
}
estimator = get_estimator(**model_config)
############
# TRAINING #
############
train_input_fn = run_classifier.input_fn_builder(
    features=train_features, seq_length=max_seq_length, is_training=True, drop_remainder=False
)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)

best_exporter = BestCheckpointsExporter(
    serving_input_receiver_fn=create_serving_input_receiver_fn(max_seq_length)
)

eval_input_fn = run_classifier.input_fn_builder(
    features=eval_features, seq_length=max_seq_length, is_training=False, drop_remainder=False
)

eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn, steps=None, exporters=best_exporter, throttle_secs=50
)

start_time = datetime.now()
tf.compat.v1.logging.info("***** Running training *****")
tf.compat.v1.logging.info("Number of examples = {}".format(len(train_examples)))
tf.compat.v1.logging.info("Batch size = {}".format(train_batch_size))
tf.compat.v1.logging.info("Num of steps = {}".format(num_train_steps))

tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)

tf.compat.v1.logging.info(
    "***** Finished training (took: {}) *****".format(datetime.now() - start_time)
)
########################
# LOAD BEST CHECKPOINT #
########################
model_config.update({"model_dir": best_checkpoint_dir})
with open(best_checkpoint_dir + "checkpoint") as file:
    checkpoint = yaml.safe_load(file)
model_config.update({"init_checkpoint": best_checkpoint_dir + checkpoint["model_checkpoint_path"]})
estimator = get_estimator(**model_config)
###################
# MODEL EVALUATION #
####################
test_input_fn = run_classifier.input_fn_builder(
    features=test_features, seq_length=max_seq_length, is_training=False, drop_remainder=False
)
predictions = estimator.predict(input_fn=test_input_fn)

y_pred = []
for prediction in predictions:
    y_pred.append(np.argmax(prediction["probabilities"]))
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
    save_path=log_dir + "confusion_matrix.png",
)
tf.compat.v1.logging.info(
    "\n" + classification_report(y_test, y_pred, target_names=label_encoder.classes_)
)
################
# EXPORT MODEL #
################
estimator.export_saved_model(
    trained_model_dir, create_serving_input_receiver_fn(max_seq_length=max_seq_length)
)
trained_model_dir = [d.as_posix() for d in Path(trained_model_dir).glob("*/") if d.is_dir()][0]
model_config.update({"trained_model_dir": trained_model_dir})
with open(pretrained_model_dir + "model_config.yml", "w") as file:
    yaml.dump(model_config, file)
###########
# CLEANUP #
###########
# Copy logs
for filepath in Path(tmp_checkpoint_dir).glob("**/events*"):
    shutil.copy(filepath, PurePath(log_dir).joinpath(filepath.name))
# Remove train dir and best checkpoint dir
shutil.rmtree(tmp_checkpoint_dir)
shutil.rmtree(best_checkpoint_dir)
# Remove pretrained model
for filepath in Path(pretrained_model_dir).glob("bert_model.ckpt.*"):
    filepath.unlink()
