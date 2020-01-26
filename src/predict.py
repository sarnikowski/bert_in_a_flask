import os
import json
import logging
import numpy as np

import data_helper

from sklearn.preprocessing import LabelEncoder
from model import create_model

logger = logging.getLogger("app.predict")


class Predict:
    """ Class to serve BERT/ALBERT predictions

    Attributes:
        model_config: Dict. Contains configuration for serving model.
        model_dir: String. Path to trained model.
        model_type: String. Expects "bert" or "albert.
        max_seq_len: Int. Max sequence length for classification.
        datahelper: data_helper.DataHelper. Class that helps tokenize input.
        model: Object. Contains the full keras model.
        encoder: Object. Label encoder.
    """

    def __init__(self):
        with open("/app/models/latest_model_config.json", "r") as filepath:
            self.model_config = json.load(filepath)
        for k, v in self.model_config.items():
            setattr(self, k, v)

        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(
            os.path.join(self.model_dir, "label_encoder.npy"), allow_pickle=True
        )
        self.model = create_model(
            model_dir=self.model_dir,
            model_type=self.model_type,
            max_seq_len=self.max_seq_len,
            n_classes=len(self.encoder.classes_),
            load_pretrained_weights=False,
        )
        self.datahelper = data_helper.DataHelper(
            model_dir=self.model_dir, model_type=self.model_type, max_seq_len=self.max_seq_len
        )
        self.model.load_weights(os.path.join(self.model_dir, "model.h5"))

    def predict(self, data):
        output = {}
        input_ids = self.datahelper.convert_data(data["x"])
        predictions = self.model.predict(input_ids)
        probabilities = np.max(predictions, axis=1)
        output["predictions"] = self.encoder.inverse_transform(
            np.argmax(predictions, axis=1)
        ).tolist()
        output["probabilities"] = probabilities.tolist()
        return output
