import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_dim,
        embedding_dim,
        hidden_size_rnn,
        number_layers_rnn,
        output_size,
        dropout=0.5,
    ):
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.hidden_size_rnn = hidden_size_rnn
        self.number_layers_rnn = number_layers_rnn
        self.output_size = output_size

        self.embedding = nn.Embedding(encoder_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size_rnn, number_layers_rnn, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size_rnn, output_size)

    def forward(self, y, hidden_states):
        y = y.unsqueeze(0)
        output = self.embedding(y)
        output = self.dropout(output)
        output, hidden_states = self.lstm(output, hidden_states)
        output = self.fc(output.squeeze(0))
        return output, hidden_states
