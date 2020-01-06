import os
import json
import logging
import numpy as np

import bert
from sklearn.preprocessing import LabelEncoder
from utils_bert import convert_data, create_model

logger = logging.getLogger("app.predict")


class Predict:
    def __init__(self):
        with open("/app/models/latest_model_config.json", "r") as filepath:
            self.model_config = json.load(filepath)
        for k, v in self.model_config.items():
            setattr(self, k, v)

        self.model = create_model(
            max_seq_len=self.max_seq_len,
            pretrained_model_dir=self.pretrained_model_dir,
            n_classes=self.n_classes,
            load_weights=False,
        )
        do_lower_case = "uncased" in self.pretrained_model_dir
        self.tokenizer = bert.bert_tokenization.FullTokenizer(
            vocab_file=os.path.join(self.pretrained_model_dir, "vocab.txt"),
            do_lower_case=do_lower_case,
        )
        self.model.load_weights(os.path.join(self.output_model_dir, "model.h5"))
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(
            os.path.join(self.output_model_dir, "label_encoder.npy"), allow_pickle=True
        )

    def predict(self, data):
        output = {}
        input_ids = convert_data(data["x"], self.tokenizer, self.max_seq_len)
        predictions = self.model.predict(input_ids)
        probabilities = np.max(predictions, axis=1)
        output["predictions"] = self.encoder.inverse_transform(
            np.argmax(predictions, axis=1)
        ).tolist()
        output["probabilities"] = probabilities.tolist()
        return output
