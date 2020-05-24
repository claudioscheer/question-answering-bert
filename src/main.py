import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math
import os
from lib import CustomDataLoader, Encoder, Decoder, SequenceToSequence


# SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_file_path = os.path.dirname(os.path.abspath(__file__))
dataset = CustomDataLoader(
    os.path.join(current_file_path, "../dataset/sample.csv"),
    split_percentages=[1, 0, 0],
)

# Define hyperparameters.
number_tokens = dataset.number_tokens
embedding_dim = 128
hidden_size_rnn = 512
number_layers_rnn = 2
dropout = 0.5

encoder = Encoder(
    number_tokens, embedding_dim, hidden_size_rnn, number_layers_rnn, dropout
)
decoder = Decoder(
    number_tokens,
    embedding_dim,
    hidden_size_rnn,
    number_layers_rnn,
    number_tokens,
    dropout,
)

model = SequenceToSequence(encoder, decoder, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
print(model)


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_model_parameters(model):,} trainable parameters.")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(
    ignore_index=dataset.token2index[dataset.pad_sentence_token]
)


def train(model, dataset, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for batch_index, (x, y) in dataset.get_batches(batch_size=16, data_type=0):
        x = torch.tensor(x).transpose(0, 1).to(device)
        y = torch.tensor(y).transpose(0, 1).to(device)

        optimizer.zero_grad()
        output = model(x, y)
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        y = y[1:]
        y = y.reshape((output.shape[0]))
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / (batch_index + 1)


def evaluate(model, dataset, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        batch_index = 0
        for batch_index, (x, y) in dataset.get_batches(batch_size=3, data_type=1):
            x = torch.tensor(x).transpose(0, 1).to(device)
            y = torch.tensor(y).transpose(0, 1).to(device)

            output = model(x, y, 0)  # turn off teacher forcing
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            y = y[1:]
            y = y.reshape((output.shape[0]))
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, y)
            epoch_loss += loss.item()
    return epoch_loss / (batch_index + 1)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 500
CLIP = 1
best_valid_loss = float("inf")
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, dataset, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, dataset, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")


def predict(model, dataset, sentence):
    tokens = dataset.get_encoded_sentence(sentence)
    tokens.insert(0, dataset.token2index[dataset.start_sentence_token])
    tokens.insert(0, dataset.token2index[dataset.start_sentence_token])

    tokens.append(dataset.token2index[dataset.end_sentence_token])
    tokens.append(dataset.token2index[dataset.end_sentence_token])

    tokens = torch.tensor([tokens]).transpose(0, 1).to(device)
    y = torch.tensor([dataset.token2index[dataset.start_sentence_token]]).to(device)

    model.eval()

    with torch.no_grad():
        answer = ""
        last_index = -1
        hidden_states = model.encoder(tokens)
        while last_index != dataset.token2index[dataset.end_sentence_token]:
            output, hidden_states = model.decoder(y, hidden_states)
            top1 = output.argmax(1)
            top1_index = top1.cpu().numpy()[0]

            last_index = top1_index
            y = torch.tensor([last_index]).to(device)

            answer += dataset.index2token[last_index] + " "
        return answer


for _ in range(100):
    answer = predict(model, dataset, "hi hi hi hi hi hi hi")
    print(answer)