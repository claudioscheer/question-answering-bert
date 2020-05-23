import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        number_words,
        embedding_dim,
        hidden_size_rnn,
        number_layers_rnn,
        dropout=0.5,
    ):
        super(Encoder, self).__init__()

        self.number_words = number_words
        self.embedding_dim = embedding_dim
        self.hidden_size_rnn = hidden_size_rnn
        self.number_layers_rnn = number_layers_rnn

        self.embedding = nn.Embedding(number_words, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size_rnn, number_layers_rnn, dropout=dropout
        )

    def forward(self, x):
        output = self.embedding(x)
        output = self.dropout(output)
        output, hidden_states = self.lstm(output)
        return hidden_states
