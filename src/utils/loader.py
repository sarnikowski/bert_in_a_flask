import os
import json
import glob
import shutil
import datetime
import bert

import custom_logger

supported_albert_models = [
    "albert_base_v2",
    "albert_large_v2",
    "albert_xlarge_v2",
    "albert_xxlarge_v2",
]

supported_bert_models = [
    "uncased_L-12_H-768_A-12",
    "uncased_L-24_H-1024_A-16",
    "cased_L-12_H-768_A-12",
    "cased_L-24_H-1024_A-16",
    "multi_cased_L-12_H-768_A-12",
    "wwm_uncased_L-24_H-1024_A-16",
    "wwm_cased_L-24_H-1024_A-16",
]

logger = custom_logger.get_logger()


def validate_model(model_name):
    """Validates the provided model name.
    Args:
        model_name: String. Name of the model. See supported models at the top.

    Returns:
        model_type: String. Either "albert" or "bert".
    """
    if model_name in supported_albert_models:
        model_type = "albert"
    elif model_name in supported_bert_models:
        model_type = "bert"
    else:
        raise ValueError(
            "Model with name:[{}] not valid. Following names are valid:{}".format(
                model_name, supported_albert_models + supported_bert_models
            )
        )
    return model_type


def fetch_model(model_name):
    """Downloads BERT/ALBERT models and outputs their location.

    Args:
        model_name: String. Name of the model. See supported models at the top.

    Returns:
        pretrained_model_dir: String. Path to pretrained model.
        model_dir: String. Path to where the trained model is saved.
        model_type: String. Either "albert" or "bert".
    """
    model_type = validate_model(model_name)
    model_dir_prefix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join("/app/models/trained", "{}_{}".format(model_dir_prefix, model_name))
    os.makedirs(model_dir, exist_ok=True)
    if model_type == "albert":
        try:
            model_file = glob.glob(
                os.path.join("/app/models/pretrained", model_name, "*", "model.ckpt-best.meta")
            )[0]
            pretrained_model_dir = os.path.dirname(model_file)
            logger.info("Located model at {}".format(pretrained_model_dir))
        except (OSError, IndexError):
            logger.info("Downloading model:[{}]".format(model_name))
            pretrained_model_dir = bert.fetch_google_albert_model(
                model_name, "/app/models/pretrained"
            )
    elif model_type == "bert":
        try:
            model_file = glob.glob(
                os.path.join(
                    "/app/models/pretrained", model_name, "bert_model.ckpt.data-00000-of-00001"
                )
            )[0]
            pretrained_model_dir = os.path.dirname(model_file)
            logger.info("Located model at {}".format(pretrained_model_dir))
        except (OSError, IndexError):
            logger.info("Downloading model:[{}]".format(model_name))
            pretrained_model_dir = bert.fetch_google_bert_model(
                model_name, "/app/models/pretrained"
            )

    return pretrained_model_dir, model_dir, model_type


def export_model_config(model_config, pretrained_model_dir, datahelper):
    """Copies necessary model files from pretrained model dir, to output model dir.

    Args:
        model_config: Dict. Keys below:
            model_dir: String. Path to where the trained model is saved.
            model_type: String. Either "albert" or "bert.
            max_seq_len: Int. Max sequence length for input examples.
        pretrained_model_dir: String. Path to pretrained model.
        datahelper: data_helper.DataHelper. Contains paths to tokenizers.
    """
    logger.info(
        "Copying model config and tokenizer from pretrained {} model, to {}".format(
            model_config["model_type"].upper(), model_config["model_dir"]
        )
    )
    if model_config["model_type"] == "albert":
        shutil.copyfile(
            datahelper.spm_file,
            os.path.join(model_config["model_dir"], os.path.basename(datahelper.spm_file)),
        )
        shutil.copyfile(
            os.path.join(pretrained_model_dir, "albert_config.json"),
            os.path.join(model_config["model_dir"], "albert_config.json"),
        )
    elif model_config["model_type"] == "bert":
        shutil.copyfile(
            datahelper.vocab_file,
            os.path.join(model_config["model_dir"], os.path.basename(datahelper.vocab_file)),
        )
        shutil.copyfile(
            os.path.join(pretrained_model_dir, "bert_config.json"),
            os.path.join(model_config["model_dir"], "bert_config.json"),
        )
    logger.info("Saving trained model configuration to {}".format(model_config["model_dir"]))
    with open(os.path.join(model_config["model_dir"], "model_config.json"), "w") as filepath:
        json.dump(model_config, filepath)
    with open("/app/models/latest_model_config.json", "w") as filepath:
        json.dump(model_config, filepath)
    logger.info("Finished exporting model files.")
