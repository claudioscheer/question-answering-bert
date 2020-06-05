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
    "eval_batch_size": 2,
    "max_seq_length": 128,
    "max_length": 128,
    "num_beams": 3,
    "early_stopping": False,
}

model = Seq2SeqModel(encoder_decoder_type="bert", encoder_decoder_name="outputs", args=model_args)
print(model.predict([sys.argv[1]]))
