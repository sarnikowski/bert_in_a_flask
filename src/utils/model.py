import os
import bert

from tensorflow import keras


def create_model(
    model_dir, model_type, max_seq_len, n_classes, load_pretrained_weights=True, summary=False,
):
    """Creates keras model with pretrained BERT/ALBERT layer.

    Args:
        model_dir: String. Path to model.
        model_type: String. Expects either "albert" or "bert"
        max_seq_len: Int. Maximum length of a classificaton example.
        n_classes: Int. Number of training classes.
        load_pretrained_weights: Boolean. Load pretrained model weights.
        summary: Boolean. Print model summary.

    Returns:
        Keras model
    """
    if model_type == "albert":
        model_ckpt = os.path.join(model_dir, "model.ckpt-best")
        model_params = bert.albert_params(model_dir)
    elif model_type == "bert":
        model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
        model_params = bert.params_from_pretrained_ckpt(model_dir)

    layer_bert = bert.BertModelLayer.from_params(model_params, name=model_type)

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype="int32", name="input_ids")
    output = layer_bert(input_ids)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=model_params["hidden_size"], activation="relu")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=n_classes, activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    if load_pretrained_weights:
        if model_type == "albert":
            bert.load_albert_weights(layer_bert, model_ckpt)
        elif model_type == "bert":
            bert.load_bert_weights(layer_bert, model_ckpt)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    if summary:
        model.summary()
    return model
