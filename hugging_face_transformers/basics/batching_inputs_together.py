from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

'''
when passing multiple sentences at the same time, the sentences have a high probability to
be of different lengths.

These differences in lengths will make it impossible to create tensors from these batches
must employ padding, which will match the smaller input to the larger one.

Truncating will only be used when the input exceeeds the maximum input length of the model

there will also need to be an attention_mask to dictate which tokens should effect the model's
results. The padding tokens should not be considered when the model is calculating its output
'''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# each model has its own padding token id
# to access it:
# print(tokenizer.pad_token_id)
sentences = [
    "I've always wanted the new GTA6!",
    "I hate this.",
]
input_ids = tokenizer(sentences, padding=True, truncation=True)
# print(input_ids)
# print(tokenizer.decode(input_ids["input_ids"][1]))
# these are the items in the input_ids that will be passed to the model
# print(input_ids["input_ids"]) # the tokens
# print(input_ids["token_type_ids"]) # in the case where there are two sentences for input,
#                                    # token_type_ids specifies which input the tokens are part of
# print(input_ids["attention_mask"]) # tells which tokens to pay attention to

'''
then pass the tokenized input to the model for the specific task
'''
# models needs tensors, so create then from the input_ids
all_ids = torch.tensor(input_ids["input_ids"])
# also make the attention mask into tensors
attention_mask = torch.tensor(input_ids["attention_mask"])
# now pass it to the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
output = model(all_ids, attention_mask=attention_mask)

'''
Alternatively, all this can be done simultaneously:

# create the pytorch tensors while the sentence is being tokenized
input_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
output = model(**input_ids)
'''

# the model only returns logits
# print(output.logits)
# run the logits through softmax to get the probability
probs = torch.nn.functional.softmax(output.logits, dim=1)
print(probs)
max_probs = torch.argmax(probs, dim=1).numpy()
# print(max_probs.shape) # (2, )
for i in range(max_probs.shape[0]):
    ans = max_probs[i]
    acc = probs[i, ans].item()
    print(f"{model.config.id2label[ans]} [{round(acc*100, 2)}%]")
