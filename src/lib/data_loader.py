import os
import pandas as pd
import torch
from nltk.tokenize import word_tokenize


class CustomDataLoader:
    def __init__(self, csv_path, validation_split=0.3):
        super(CustomDataLoader, self).__init__()

        self.start_sentence_token = "<SOS>"  # start of sentence
        self.end_sentence_token = "<EOF>"  # end of sentence
        self.pad_sentence_token = "<PAD>"  # end of sentence

        self.csv = pd.read_csv(csv_path, header=0)
        self.word2index = {
            self.start_sentence_token: 0,
            self.end_sentence_token: 1,
            self.pad_sentence_token: 2,
        }
        self.index2word = {
            0: self.start_sentence_token,
            1: self.end_sentence_token,
            2: self.pad_sentence_token,
        }
        self.word_count = {}
        self.number_words = 3

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

    def tokenize_sentence(self, sentence):
        return word_tokenize(sentence)

    def add_sentence(self, sentence):
        for word in self.tokenize_sentence(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.number_words
            self.index2word[self.number_words] = word
            self.word_count[word] = 1
            self.number_words += 1
        else:
            self.word_count[word] += 1

    def get_encoded_sentence(self, sentence, reverse=False):
        encoded = []
        for word in self.tokenize_sentence(sentence):
            encoded.append(self.word2index[word])
        if reverse:
            encoded = encoded[::-1]
        return encoded

    def pad_batch_sequence(self, x, y):
        longer_x = len(max(x, key=len))
        longer_y = len(max(y, key=len))
        for i in range(len(x)):
            # Append pad token.
            for _ in range(longer_x - len(x[i])):
                x[i].append(self.word2index[self.pad_sentence_token])
            for _ in range(longer_y - len(y[i])):
                y[i].append(self.word2index[self.pad_sentence_token])
            # Insert SOS token.
            x[i].insert(0, self.word2index[self.start_sentence_token])
            y[i].insert(0, self.word2index[self.start_sentence_token])
            # Append EOF token.
            x[i].append(self.word2index[self.end_sentence_token])
            y[i].append(self.word2index[self.end_sentence_token])
        return x, y

    def get_batches(self, batch_size, drop_last=False, validation=False):
        indexes = self.validation_indexes if validation else self.train_indexes
        batch_index = -1
        count = 0
        while count < len(indexes) and batch_size <= len(indexes):
            x = []
            y = []
            for i in range(batch_size):
                row = self.csv.iloc[indexes[i + count]]
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
                row = self.csv.iloc[indexes[i + count]]
                x.append(self.get_encoded_sentence(row["question"]))
                y.append(self.get_encoded_sentence(row["answer"]))
            batch_index += 1
            yield batch_index, self.pad_batch_sequence(x, y)


if __name__ == "__main__":
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    dataset = CustomDataLoader(
        os.path.join(current_file_path, "../../dataset/supervised.csv")
    )
    for x, y in dataset.get_batches(3):
        print(x, y)
