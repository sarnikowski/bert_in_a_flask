import yaml
import logging
import numpy as np

from bert import tokenization
from tensorflow.contrib import predictor
from sklearn.preprocessing import LabelEncoder

from utils_bert import create_examples, convert_examples_to_features

logger = logging.getLogger("app.predict")


class Predict:
    def __init__(self):
        with open("/app/models/pretrained/model_config.yml") as file:
            self.model_config = yaml.safe_load(file)

        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(
            "/app/models/pretrained/label_encoder.npy", allow_pickle=True
        )

        self.label_list = [str(num) for num in range(self.model_config["num_labels"])]
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.model_config["vocab_file"],
            do_lower_case=self.model_config["do_lower_case"],
        )

        self.predict_fn = predictor.from_saved_model(self.model_config["trained_model_dir"])

    def predict(self, data):
        output = {}
        predictions = []
        prediction_probs = []
        X = data["X"]
        print(X)
        test_examples = create_examples(X, [0] * len(X))
        print(test_examples)
        test_features = convert_examples_to_features(
            test_examples, self.label_list, self.model_config["max_seq_length"], self.tokenizer
        )
        print(test_features)
        for test_feature in test_features:
            test_feature = test_feature.__dict__
            test_feature.pop("is_real_example", "None")
            test_feature["label_ids"] = test_feature.pop("label_id")
            test_feature = {key: [value] for key, value in test_feature.items()}
            # Perform prediction
            prediction = self.predict_fn(test_feature)
            predictions.append(
                self.encoder.inverse_transform([np.argmax(prediction["probabilities"])])[0]
            )
            prediction_probs.append(float(np.max(prediction["probabilities"][0])))
        output["predictions"] = predictions
        output["prediction_probs"] = prediction_probs
        return output
