import json
from transformers import BertTokenizer
import os


def get_dataset_generator(path):
    with open(path, "r") as file:
        data = file.read()
    data = json.loads(data)
    data = data["data"]
    for x in data:
        for y in x["paragraphs"]:
            context = y["context"]
            for z in y["qas"]:
                question = z["question"]
                for r in z["answers"]:
                    answer = r["text"]
                    answer_start = r["answer_start"]
                    yield (context, question, answer, answer_start)


def tokenize_batch(batch, labels, tokenizer):
    batch = tokenizer.batch_encode_plus(
        batch, max_length=512, return_tensors="pt", pad_to_max_length=True
    )
    return batch, labels


def get_batches(dataset_generator, batch_size, tokenizer):
    while True:
        batch = []
        labels = []
        for _ in range(batch_size):
            try:
                (context, question, answer, answer_start) = next(dataset_generator)
            except:
                if len(batch) > 0:
                    yield tokenize_batch(batch, labels, tokenizer)
                return
            batch.append((context, question))
            labels.append(
                (
                    tokenizer.encode(
                        answer, return_tensors="pt", pad_to_max_length=True
                    ),
                    answer_start,
                )
            )
        yield tokenize_batch(batch, labels, tokenizer)


if __name__ == "__main__":
    path = os.path.basename(os.path.dirname(__file__))
    dataset = get_dataset_generator(
        os.path.join(path, "../../dataset/squad/dev-v1.1.json")
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for x, y in get_batches(dataset, 64, tokenizer):
        print(x)
        # print(y)
        break
