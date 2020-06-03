import os
import logging
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel


def get_dataset(file_path):
    train = []
    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(file_path, file_path), "r") as file:
        train = file.readlines()
    data = []
    i = 0
    while i < len(train):
        data.append([train[i], train[i + 1]])
        i += 2
    df = pd.DataFrame(data, columns=["input_text", "target_text"])
    return df


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = get_dataset("../../dataset/twitter-dev/twitter_en.train.txt")
eval_df = get_dataset("../../dataset/twitter-dev/twitter_en.eval.txt")

model_args = {
    "fp16": False,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "num_train_epochs": 100,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": True,
    "evaluate_generated_text": True,
    "evaluate_during_training": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "save_best_model": True,
    "max_length": 128,
    "num_beams": 3,
    "early_stopping": False,
    "learning_rate": 1e-4,
}

model = Seq2SeqModel("bert", "bert-base-cased", "bert-base-cased", args=model_args)

model.train_model(train_df, eval_data=eval_df, matches=count_matches)

print("Evaluating model:")
print(model.eval_model(eval_df, matches=count_matches))

# print("Generating prediction:")
# output = model.predict(["bre takin shots"])[0]
# print(output)
# print(len(output))
