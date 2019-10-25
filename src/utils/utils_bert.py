import tensorflow as tf

from bert import optimization
from bert import run_classifier
from bert import modeling


def create_examples(X, y):
    """ Creates BERT compliant data examples

    Arguments
    ---------
    X: string
        Text body to predict
    y: int
        target to predict

    Returns
    -------
    examples: bert.run_classifier.InputExample object
        data objects
    """
    examples = []
    for i, (text, target) in enumerate(zip(X, y)):
        examples.append(
            run_classifier.InputExample(guid=i, text_a=text, text_b=None, label=str(target))
        )
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = run_classifier.convert_single_example(
            10, example, label_list, max_seq_length, tokenizer
        )

        features.append(feature)
    return features


def create_serving_input_receiver_fn(max_seq_length):
    """ Builds a serving_inputer_receiver_fn

    Arguments
    ---------
    max_seq_length: int
        Specifies the sequence length

    Returns
    -------
    serving_input_receiver_fn()
    """

    def serving_input_receiver_fn():
        """ Creates an serving_input_receiver_fn for BERT"""
        unique_ids = tf.placeholder(tf.int32, [None], name="unique_ids")
        input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name="input_ids")
        input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name="segment_ids")
        label_ids = tf.placeholder(tf.int32, [None], name="label_ids")
        return tf.estimator.export.build_raw_serving_input_receiver_fn(
            {
                "unique_ids": unique_ids,
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids,
            }
        )()

    return serving_input_receiver_fn


def get_estimator(**kwargs):
    """ Builds BERT model_fn and tf.estimator.RunConfig from a param set
    and returns an estimator object

    Arguments
    ---------
    **kwargs: **kwargs
        param dictionary

    Returns
    -------
    estimator: tf.estimator.Estimator()
    """
    run_config = tf.estimator.RunConfig(
        model_dir=kwargs.get("model_dir"),
        save_summary_steps=kwargs.get("save_summary_steps"),
        save_checkpoints_steps=kwargs.get("save_checkpoints_steps"),
        keep_checkpoint_max=kwargs.get("keep_checkpoint_max"),
    )

    bert_config = modeling.BertConfig.from_json_file(kwargs.get("bert_config_path"))
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=kwargs.get("num_labels"),
        init_checkpoint=kwargs.get("init_checkpoint"),
        learning_rate=kwargs.get("learning_rate"),
        num_train_steps=kwargs.get("num_train_steps"),
        num_warmup_steps=kwargs.get("num_warmup_steps"),
        use_tpu=False,
        use_one_hot_embeddings=False,
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params={"batch_size": kwargs.get("train_batch_size")}
    )
    return estimator


def model_fn_builder(
    bert_config,
    num_labels,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for Estimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        (total_loss, per_example_loss, logits, probabilities) = run_classifier.create_model(
            bert_config,
            is_training,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            num_labels,
            use_one_hot_embeddings,
        )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, scaffold=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example
                )
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {"eval_accuracy": accuracy, "eval_loss": loss}

            eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, eval_metric_ops=eval_metrics, scaffold=scaffold_fn
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions={"probabilities": probabilities}, scaffold=scaffold_fn
            )
        return output_spec

    return model_fn
