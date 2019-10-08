import yaml
import numpy as np

from bert import tokenization
from tensorflow.contrib import predictor
from sklearn.preprocessing import LabelEncoder

from utils_bert import create_examples, convert_examples_to_features
from utils_preprocess import clean_string_bert


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

        self.predict_fn = predictor.from_saved_model(
            self.model_config["trained_model_dir"]
        )
#TODO Refactor this once serving_input_fn_receiver is fixed 
    def predict(self, data):
        output = {}
        # DEFINE INPUT STRING HERE
        input_string = None
        input_string = clean_string_bert(input_string)

        test_examples = create_examples([input_string], [0])
        test_features = convert_examples_to_features(
            test_examples,
            self.label_list,
            self.model_config["max_seq_length"],
            self.tokenizer,
        )
        # Convert features to be single pass compatible
        for test_feature in test_features:
            features = test_feature

        features = features.__dict__
        features.pop("is_real_example", "None")
        features["label_ids"] = features.pop("label_id")
        features = {key: [value] for key, value in features.items()}
        # Perform prediction
        prediction = self.predict_fn(features)
        y_pred = prediction["probabilities"]

        y_pred_max = np.argmax(y_pred)
        y_pred_probability = np.max(y_pred)

        label_predicted = self.encoder.inverse_transform([y_pred_max])

        output["routing_label"] = label_predicted[0]
        output["routing_label_probability"] = str(y_pred_probability)
        return output
