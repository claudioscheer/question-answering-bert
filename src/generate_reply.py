import logging
import sys
from lib import Seq2SeqModel


if len(sys.argv) < 2:
    print("ERROR: You need to inform a text.")
    sys.exit()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model = Seq2SeqModel(encoder_decoder_type="bert", encoder_decoder_name="outputs")
print(model.predict([sys.argv[1]]))
