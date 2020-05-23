import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize


class CustomDataLoader:
    def __init__(self, csv_path, split_percentages=[0.7, 0.15, 0.15]):
        assert sum(split_percentages) == 1, "The sum of `split_percentages` must be 1."
        assert os.path.exists(csv_path), "The argument `csv_path` is invalid."

        self.dataset = pd.read_csv(csv_path, header=0)

        # Split dataset into training, validation and testing.
        csv_indexes = list(range(len(self)))
        np.random.shuffle(csv_indexes)
        validation_count = int(len(csv_indexes) * split_percentages[1])
        testing_count = int(len(csv_indexes) * split_percentages[2])

        self.validation_indexes = csv_indexes[:validation_count]
        self.testing_indexes = csv_indexes[
            validation_count : validation_count + testing_count
        ]
        self.training_indexes = csv_indexes[validation_count + testing_count :]

        assert sum(
            [
                len(self.validation_indexes),
                len(self.testing_indexes),
                len(self.training_indexes),
            ]
        ) == len(csv_indexes), "An error occured while splitting the dataset."

        self.start_sentence_token = "<SOS>"  # start of sentence
        self.end_sentence_token = "<EOF>"  # end of sentence
        self.pad_sentence_token = "<PAD>"  # sentence pad

        self.word2token = {
            self.start_sentence_token: 0,
            self.end_sentence_token: 1,
            self.pad_sentence_token: 2,
        }
        self.index2token = {
            0: self.start_sentence_token,
            1: self.end_sentence_token,
            2: self.pad_sentence_token,
        }
        self.token_count = {}
        self.number_tokens = 3

        # Create the dictionary of words and indexes.
        for x in range(len(self)):
            row = self.dataset.iloc[x]
            self.add_sentence(row["question"])
            self.add_sentence(row["answer"])

    def __len__(self):
        return len(self.dataset)

    def tokenize_sentence(self, sentence):
        return word_tokenize(sentence)

    def add_sentence(self, sentence):
        for token in self.tokenize_sentence(sentence):
            self.add_token(token)

    def add_token(self, token):
        if token not in self.word2token:
            self.word2token[token] = self.number_tokens
            self.index2token[self.number_tokens] = token
            self.token_count[token] = 1
            self.number_tokens += 1
        else:
            self.token_count[token] += 1

    def get_encoded_sentence(self, sentence, reverse=False):
        encoded = []
        for word in self.tokenize_sentence(sentence):
            encoded.append(self.word2token[word])
        if reverse:
            # Based on https://arxiv.org/abs/1409.3215.
            encoded = encoded[::-1]
        return encoded

    def pad_batch_sequence(self, x, y):
        assert len(x) == len(y), "`x` and `y` must be the same length."

        longer_x = len(max(x, key=len))
        longer_y = len(max(y, key=len))
        for i in range(len(x)):
            # Append pad token.
            for _ in range(longer_x - len(x[i])):
                x[i].append(self.word2token[self.pad_sentence_token])
            for _ in range(longer_y - len(y[i])):
                y[i].append(self.word2token[self.pad_sentence_token])
            # Insert SOS token.
            x[i].insert(0, self.word2token[self.start_sentence_token])
            y[i].insert(0, self.word2token[self.start_sentence_token])
            # Append EOF token.
            x[i].append(self.word2token[self.end_sentence_token])
            y[i].append(self.word2token[self.end_sentence_token])
        return x, y

    def get_data_indices(self, data_type):
        """
            data_type: 0 for training data; 1 for validation data; 2 for testing data.
        """
        if data_type == 0:
            return self.training_indexes
        elif data_type == 1:
            return self.validation_indexes
        elif data_type == 2:
            return self.testing_indexes
        else:
            raise Exception("Invalid `data_type`.")

    def get_batches(self, batch_size, data_type, drop_last=False):
        """
            batch_size: The batch size used as input in the model.
            data_type: 0 for training data; 1 for validation data; 2 for testing data.
            drop_last: When True, the last batch may not be the same size as `batch_size`.
        """
        indexes = self.get_data_indices(data_type)
        batch_index = -1
        count = 0
        while count < len(indexes) and batch_size <= len(indexes):
            x = []
            y = []
            for i in range(batch_size):
                row = self.dataset.iloc[indexes[i + count]]
                x.append(self.get_encoded_sentence(row["question"]))
                y.append(self.get_encoded_sentence(row["answer"]))
            batch_index += 1
            yield batch_index, self.pad_batch_sequence(x, y)
            count += batch_size
            if count + batch_size > len(indexes):
                break

        if not drop_last and count < len(indexes):
            x = []
            y = []
            for i in range(len(indexes) - count):
                row = self.dataset.iloc[indexes[i + count]]
                x.append(self.get_encoded_sentence(row["question"]))
                y.append(self.get_encoded_sentence(row["answer"]))
            batch_index += 1
            yield batch_index, self.pad_batch_sequence(x, y)
