import logging
import sys
from lib import Seq2SeqModel


if len(sys.argv) < 2:
    print("ERROR: You need to inform a text.")
    sys.exit()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = {
    "fp16": False,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "train_batch_size": 16,
    "num_train_epochs": 64,
    "max_length": 256,
    "learning_rate": 1e-4,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "save_best_model": False,
    "save_steps": 100000,
    "num_beams": 5,
    "gradient_accumulation_steps": 1,
}

model = Seq2SeqModel(encoder_decoder_type="bert", encoder_decoder_name="outputs", args=model_args)
print(model.predict([sys.argv[1]]))
