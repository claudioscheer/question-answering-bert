import os
import logging
import pandas as pd
from lib import Seq2SeqModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

file_path = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(file_path, "../../dataset/input-target.csv"))
train_df.drop(train_df.columns[[0]], axis=1, inplace=True)
train_df.dropna(subset=["input_text", "target_text"], inplace=True)

model_args = {
    "fp16": False,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 1,
    "num_train_epochs": 16,
    "max_length": 128,
    "num_beams": 3,
    "early_stopping": False,
    "learning_rate": 3e-5,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "save_best_model": False,
    "gradient_accumulation_steps": 1,
}

model = Seq2SeqModel(encoder_type="bert", encoder_name="bert-base-cased", decoder_name="bert-base-cased", args=model_args)
model.train_model(train_df)
