import logging
import sys
from torchtext.data.metrics import bleu_score
from transformers import BertTokenizer
from lib import Seq2SeqModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

dataset = [
    ("that is fine.", "Thank you."),
    ("Are you coming back to work?  Or are you going home?", "i am coming back to work.   do you need me to take you to pick up your car?"),
    ("you look so handsome today!", "you are making me blush."),
    ("hey what does your middle initial  H stand for", "Harris"),
    (
        "Nella here is a list of products that we would launch on EOL in NETCO.  Assume the the products listed under the Financial heading are launched on day 1 and the the products listed under the Physical heading are launched at a later date.",
        "JOHN, NO LIST ATTACHED.  PLEASE RESEND. THANKS, NELLA",
    ),
    ("i don't even know your salary now.", "175,000"),
    (
        """Zuf:

        Howdy!  I hope all is well.

        What is your fax# @ work?

        Brother Smith""",
        "403-974-6706",
    ),
    ("Friendly reminder.", "I'll be there"),
]

# model = Seq2SeqModel(encoder_decoder_type="bert", encoder_decoder_name="outputs")
# outputs = model.predict([x[0] for x in dataset])

outputs = [
    "Thank you. all the best.",
    "i am coming back to work. do you need me to take you to pick up your car?",
    "you are making me blush.",
    "Harris",
    "ETS - ASAP is still active. I will let you know what the status of the EPMI and NNG pipelines. Thanks, Melba",
    "haha. i am going to buy a car for you. why don't you call me about that?..",
    "a secret for me too? What time are you going to be in town? I will be out of town on Friday. Chris",
    "Thanks. of course you'll be taking some time off. too much risk.",
]
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

candidate_corpus = [tokenizer.tokenize(x[1]) for x in dataset]
references_corpus = [[tokenizer.tokenize(x)] for x in outputs]
bleu = bleu_score(candidate_corpus, references_corpus)
print(bleu)
