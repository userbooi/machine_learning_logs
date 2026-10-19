from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

from pprint import pprint

'''
Although the pipeline function gives a simple way to go from raw text to prediction for
any task, the process can be done and implemented step by step.

the pipeline function has steps:

Tokenizer -> Model -> Postprocessing

Tokenizers:
    - splits the input into individual tokens
    - these tokens are numerical, pointing to an index in the model's vocabulary
    - special <SOS> tokens and <EOS> tokens

Model:
    - feeds the tokens into transformers
        - depending on the model, the type of transformer architecture will be different
        - eg. encoder-only (BERT), decoder-only (GPT2), encoder-decoder (BART)
    - returns only the logits without the head (top) layers

Postprocessing:
    - returns the predictions of the model in human-readable fashion
    - takes the logits and pass them through an output layer

these can all be implemented separately with hugging face
'''

# the information for a specific model (this is used for all the steps)
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# ====================== Tokenizer ==============================
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for the new GTA6 my whole life",
    "I've wanted the new GTA6 for my whole life!"
]
inputs = tokenizer(
    raw_inputs,
    padding=True, # adds filling for the shorter sentences
    truncation=True, # removes extra tokens from a longer sentence
    return_tensors="pt", # pt means pytorch
    # the attention mask shows which values to pay attention to (anything but padding)
    # the token ids are the corresponding index in the "vocabulary"
)

# pprint(inputs)

# ====================== AutoModel ==============================
# model = AutoModel.from_pretrained(checkpoint)
#
# # the model only includes the body, which means the top layers (head) is removed
# outputs = model(**inputs) # the ** unpacks the dictionary. So this is equivalent to:
#                           # model(attention_mask=..., input_ids=..., token_type_ids=...)
#
# print(outputs.last_hidden_state.shape) # (batch_size, sequence_length, hidden_size)

# ====================== AutoModelForSequenceClassification ==============================
# this is the same thing, but it has a classification head for a specific task
# in this case, it is sequence classification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# now it will have two outputs for each sentence
# however, since these are logits, it is not yet probabilities
# print(outputs.logits)

# ====================== PostProcessing ==============================
# this step turns the logits into something useful, like probabilities (predictions)
predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
# print(predictions)

# get the index with the highest probabilities
predicted_ids = torch.argmax(predictions, dim=1)
print(predicted_ids)

for id in predicted_ids.numpy():
    print(model.config.id2label[id])

