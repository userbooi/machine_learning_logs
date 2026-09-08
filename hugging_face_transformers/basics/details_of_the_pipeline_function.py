from transformers import AutoTokenizer, TFAutoModel

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

# ====================== Tokensizer ==============================
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for the new GTA6 my whole life",
    "I hate this so much"
]
inputs = tokenizer(
    raw_inputs,
    padding=True, # adds filling for the shorter sentences
    truncation=True, # removes extra tokens from a longer sentence
    return_tensors="pt", # uses tensorflow. "tf" is for pytorch
    #
)
# print(inputs)

