import logging
from lib import Seq2SeqModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = {
    "fp16": False,
    "gradient_accumulation_steps": 4,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "train_batch_size": 4,
    "eval_batch_size": 1,
    "num_train_epochs": 16,
    "max_length": 128,
    "num_beams": 3,
    "early_stopping": False,
    "learning_rate": 1e-4,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "save_best_model": False,
}

model = Seq2SeqModel(encoder_decoder_type="bert", encoder_decoder_name="outputs", args=model_args)
print(model.predict(["I like how I am"]))
