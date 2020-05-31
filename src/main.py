from transformers import BertTokenizer, BertForQuestionAnswering
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert-large-uncased-whole-word-masking-finetuned-squad
# model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

question, text = (
    "Where did Super Bowl 50 take place?",
    'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.',
)
encoding = tokenizer.encode_plus(question, text)
input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]

start_scores, end_scores = model(
    torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids])
)
all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(all_tokens)
answer = " ".join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1])
print(answer)
