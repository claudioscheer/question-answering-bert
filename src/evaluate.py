import os
import pandas as pd
from torchtext.data.metrics import bleu_score
from transformers import BertTokenizer


file_path = os.path.dirname(os.path.abspath(__file__))
evaluation_df = pd.read_csv(os.path.join(file_path, "../dataset/input-target-256-evaluation.csv"))

dataset = []
outputs = []
for i, x in evaluation_df.iterrows():
    dataset.append((str(x["input_text"]), str(x["target_text"])))
    outputs.append(str(x["generated_reply"]))

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

candidate_corpus = [tokenizer.tokenize(x[1]) for x in dataset]
references_corpus = [[tokenizer.tokenize(x)] for x in outputs]
bleu = bleu_score(candidate_corpus, references_corpus)
print(bleu)
