import os
import logging
import numpy as np
import pandas as pd
import bert
import sentencepiece as spm
import params_flow as pf

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class DataHelper(object):
    """Converts data using BERT/ALBERT tokenization/encoding strategies.

    Attributes:
        model_dir: String. Path to model directory, where either vocab file (BERT)
            or sentencepiece model file (ALBERT) is located.
        model_type: String. Either "albert" or "bert".
        max_seq_len: Int. Max sequence length for input examples.
        vocab_file: String. Path to BERT vocab.txt file.
        bert_tokenizer: bert.bert_tokenization.FullTokenizer.
        spm_file: String. Path to ALBERT spm model.
        spm_model: sentencepiece.SentencePieceProcessor().
    """

    def __init__(self, model_dir, model_type, max_seq_len):
        self.model_dir = model_dir
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.vocab_file = None
        self.spm_file = None
        if model_type == "albert":
            self.spm_model, self.spm_file = self._load_albert_sentencepiece_model()
        elif model_type == "bert":
            self.bert_tokenizer, self.vocab_file = self._load_bert_tokenizer()

    def convert_data(self, x):
        """Converts data to ALBERT/BERT id encoded input examples.

        Args:
            x: List. List of data examples to convert.

        Returns:
            List. Id encoded input examples.
        """
        if self.model_type == "albert":
            return self._id_encode_albert(x)
        elif self.model_type == "bert":
            return self._id_encode_bert(x)

    def get_stackoverflow_data(self, model_dir, test_size, random_state=None):
        """Retrieves the stackoverflow dataset and preprocesses this by splitting and tokenizing.
        (https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv)

        Args:
            model_dir: String. Path to where the trained model is saved.
            test_size: Float. Fraction of the dataset to use for test.
            random_state: Int. Seed for train/test split.

        Returns:
            x_train: List. Training examples.
            x_test: List. Test exmaples.
            y_train: List. Encoded labels for training set.
            y_test: List. Encoded labels for test set.
            label_encoder: sklearn.preprocessing.LabelEncoder()
        """
        if not os.path.exists("/app/data/stack-overflow-data.csv"):
            logger.info("Downloading stackoverflow data.")
            pf.utils.fetch_url(
                "https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv",
                fetch_dir="/app/data/",
            )
            logger.info("Finished downloading data.")
        df = pd.read_csv("/app/data/stack-overflow-data.csv")
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(df["tags"])
        np.save(os.path.join(model_dir, "label_encoder.npy"), label_encoder.classes_)

        y_train = label_encoder.transform(df_train["tags"])
        y_test = label_encoder.transform(df_test["tags"])

        logger.info("Converting data to {} format...".format(self.model_type.upper()))
        x_train = self.convert_data(df_train["post"])
        x_test = self.convert_data(df_test["post"])
        logger.info("Finished converting data to {} format".format(self.model_type.upper()))

        logger.info("x_train shape: {}".format(x_train.shape))
        logger.info("y_train shape: {}".format(y_train.shape))
        return x_train, x_test, y_train, y_test, label_encoder

    def _id_encode_albert(self, x):
        """Creates BERT encoded input_ids. Truncates/pads if necessary and adds
        special character [CLS] and [SEP].

        Args:
            x: List. Data examples.

        Returns:
            input_ids: List. ALBERT id encoded examples.
        """
        cls_id = [self.spm_model.PieceToId("[CLS]")]
        sep_id = [self.spm_model.PieceToId("[SEP]")]
        input_ids = [
            bert.albert_tokenization.encode_ids(self.spm_model, input_token_example) for input_token_example in x
        ]
        input_ids = self._truncate_pad(input_ids, cls_id, sep_id)
        return input_ids

    def _id_encode_bert(self, x):
        """Creates BERT encoded input_ids. Truncates/pads if necessary and adds
        special character [CLS] and [SEP].

        Args:
            x: List. Data examples.

        Returns:
            input_ids: List. BERT id encoded examples.
        """
        input_tokens = map(self.bert_tokenizer.tokenize, x)
        input_ids = list(map(self.bert_tokenizer.convert_tokens_to_ids, input_tokens))
        cls_id = self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"])
        sep_id = self.bert_tokenizer.convert_tokens_to_ids(["[SEP]"])
        input_ids = self._truncate_pad(input_ids, cls_id, sep_id)
        return input_ids

    def _truncate_pad(self, input_ids, cls_id, sep_id):
        """Prepends [CLS] and appends [SEP]. Truncates/pads if necessary based on max_seq_len.

        Args:
            input_ids: List. BERT/ALBERT id encoded input examples.
            cls_id: List. [CLS] id encoded.
            sep_id: List. [SEP] id encoded.

        Returns:
            input_ids: List. List of id encoded input examples.
        """
        input_ids = list(
            map(
                lambda ids: cls_id + ids + sep_id
                if len(ids) <= self.max_seq_len - 2
                else cls_id + ids[: self.max_seq_len - 2] + sep_id,
                input_ids,
            )
        )
        input_ids = map(lambda tids: tids + [0] * (self.max_seq_len - len(tids)), input_ids)
        input_ids = np.array(list(input_ids))
        return input_ids

    def _load_bert_tokenizer(self):
        """Construct and loads a BERT tokenizer from vocab file.

        Returns:
            tokenizer: bert.bert_tokenization.FullTokenizer.
        """
        vocab_file = os.path.join(self.model_dir, "vocab.txt")
        do_lower_case = "uncased" in self.model_dir
        tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        return tokenizer, vocab_file

    def _load_albert_sentencepiece_model(self):
        """Construct and loads a ALBERT sentencepiece model from vocab file.

        Returns:
            spm_model: sentencepiece.SentencePieceProcessor()
        """
        spm_file = os.path.join(self.model_dir, "30k-clean.model")
        spm_model = spm.SentencePieceProcessor()
        spm_model.load(spm_file)
        return spm_model, spm_file
