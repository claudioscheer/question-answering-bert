import os
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
import torch
from lib.dataset import get_dataset_generator, get_batches


current_path = os.path.basename(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
model.train()

train_dataset_generator = get_dataset_generator(
    os.path.join(current_path, "../dataset/squad/train-v1.1.json")
)
test_dataset_generator = get_dataset_generator(
    os.path.join(current_path, "../dataset/squad/dev-v1.1.json")
)

n_epochs = 10
for _ in range(n_epochs):
    for x, y in get_batches(test_dataset_generator, 16, tokenizer):
        pass
