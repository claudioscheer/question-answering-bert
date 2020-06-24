import os
import logging
import pandas as pd
from lib import Seq2SeqModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

file_path = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(file_path, "../../dataset/input-target-256-dev.csv"))
train_df.drop(train_df.columns[[0]], axis=1, inplace=True)
train_df.dropna(subset=["input_text", "target_text"], inplace=True)

model_args = {
    "fp16": False,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "train_batch_size": 8,
    "num_train_epochs": 128,
    "max_length": 256,
    "learning_rate": 3e-5,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "save_best_model": False,
    "save_steps": 100000,
    "num_beams": 3,
    "gradient_accumulation_steps": 1,
    "use_multiprocessing": False,
}

model = Seq2SeqModel(encoder_type="bert", encoder_name="bert-base-cased", decoder_name="bert-base-cased", args=model_args)
model.train_model(train_df)
