import os
import logging
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

dataset = []
file_path = os.path.basename(os.path.dirname(__file__))
with open(
    os.path.join(file_path, "../../dataset/twitter-dev/twitter_en.txt"), "r"
) as file:
    dataset = file.readlines()

train_data = []
i = 0
while i < len(dataset):
    train_data.append([dataset[i], dataset[i + 1]])
    i += 2

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = [
    [
        "thinks you're mr. moon's son!",
        "i'm okay with that. it had nothing to do with a bright orange astros jersey i was wearing 😉",
    ],
]
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = {
    "fp16": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "train_batch_size": 1,
    "num_train_epochs": 50,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_generated_text": True,
    "evaluate_during_training": False,
    "evaluate_during_training_verbose": False,
    "use_multiprocessing": False,
    "save_best_model": False,
    "max_length": 512,
}

model = Seq2SeqModel("bert", "bert-base-cased", "bert-base-cased", args=model_args)


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


model.train_model(train_df, eval_data=eval_df, matches=count_matches)

print("Evaluating model:")
print(model.eval_model(eval_df, matches=count_matches))

print("Generating prediction:")
print(model.predict(["bre takin shots"]))
