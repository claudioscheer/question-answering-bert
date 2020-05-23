import os
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torchtext.data import Field, BucketIterator


class CustomDataLoader:
    def __init__(self, csv_path, validation_split=0.3):
        super(CustomDataLoader, self).__init__()

        self.start_sentence_token = "<SOS>"  # start of sentence
        self.end_sentence_token = "<EOF>"  # end of sentence

        self.csv = pd.read_csv(csv_path, header=0)
        self.word2index = {self.start_sentence_token: 0,
                           self.end_sentence_token: 1}
        self.index2word = {0: self.start_sentence_token,
                           1: self.end_sentence_token}
        self.word_count = {}
        self.number_words = 2

        csv_indexes = list(range(len(self)))
        split_dataset_index = int(len(csv_indexes) * validation_split)
        self.validation_indexes = csv_indexes[:split_dataset_index]
        self.train_indexes = csv_indexes[split_dataset_index:]

        # Create the dictionary of words and indexes.
        for x in range(len(self)):
            row = self.csv.iloc[x]
            self.add_sentence(row["question"])
            self.add_sentence(row["answer"])

    def __len__(self):
        return len(self.csv)

    def add_sentence(self, sentence):
        for word in word_tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.number_words
            self.index2word[self.number_words] = word
            self.word_count[word] = 1
            self.number_words += 1
        else:
            self.word_count[word] += 1

    def get_encoded_sequence(self, sentence, reverse=False):
        encoded = []
        for word in word_tokenize(sentence):
            encoded.append(self.word2index[word])
        if reverse:
            encoded = encoded[::-1]
        encoded.insert(0, self.word2index[self.start_sentence_token])
        encoded.append(self.word2index[self.end_sentence_token])
        return encoded

    def get_batches(self, validation=False):
        indexes = self.validation_indexes if validation else self.train_indexes
        for x in indexes:
            row = self.csv.iloc[x]
            x = self.get_encoded_sequence(row["question"])
            y = self.get_encoded_sequence(row["answer"])
            yield x, y


if __name__ == "__main__":
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    dataset = CustomDataLoader(os.path.join(
        current_file_path, "../../dataset/supervised.csv"))
    for x, y in dataset.get_batches():
        print(x, y)
        break
