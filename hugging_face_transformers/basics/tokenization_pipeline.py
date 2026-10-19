from transformers import AutoTokenizer

'''
There are three main types of tokenization:
    1. Word-based
    2. Character-based and
    3. subword-based

Word-based tokenization:
    - most basic tokenization
        - each word in a sentence will receive its own token
        - each token is meaningful
    - however, vocabularies can get very large
        - as there are many words in languages
    - one may create a vocabulary with the 10000 most common words for example
        - However, there may be a lot of words with <UKN> tokens
        - sentences may be difficult to full represent
    - similar words have loss of meaning

Character-based tokenization:
    - Solves Word-based's large vocabulary issue, as there is only a fixed number
      characters present in each language
    - however, tokenized sequences can get very long
        - each character becomes a token
    - each token itself holds diminished meaning

Subword-based tokenization
    - a middle ground between word-based and subword based
        - reduces vocabulary size but balances tokenized sequence length
        - gives meaning to similar words, while tokens have moderate meaning
    - the general rule for subword tokenization
        - frequently used words should not be split into smaller subwords
        - rare words should be decomposed into meaningful subwords
        - for example:
            - dog -/-> 'd' 'o' 'g'
            - dogs --> 'dog' 's'
            - this way, the model recognizes that dogs is still related to dog,
              but with added information of the plural form
        - another example:
            - tokenization --> 'token' 'ization'
            - 'token' is the base, and 'ization' as the completing part
            - this way, 'token', 'tokens', 'tokenizing', and 'tokenization' will
              have similar meaning
            - 'tokenization', 'modernization', and 'immunization' will be learned to
              be used in similar syntactic situations
    - there will generally be ways to recognize if a token is start-of-word or completion-of-word
        - for example, BERT uses '##' as completion, and '<nothing>' as start of word

There exists a lot of subword tokenizing algorithms:
    - WordPiece
        - used by BERT and DistilBERT
    - Unigram
        - used by XLNet, ALBERT
    - Byte-Pair Encoding
        - GPT-2, RoBERTa
        
        
This is the general pipeline for tokenization:
    Raw Text -> Tokens -> Special Tokens -> Input IDs
'''

'''
individual steps
'''
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# # tokenizer.save_pretrained("./tokenizer/bert_tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer/bert_tokenizer")
# tokens = tokenizer.tokenize("Let's try to tokenize!") # only tokenizes the words
#                                                       # since this is bert-base-uncased, it
#                                                       # will also lower-case all words first
# print(tokens)
# input_ids = tokenizer.convert_tokens_to_ids(tokens) # turns the tokens into indexes
# print(input_ids)
# # prepare_for_model doesn't exist anymore
# # final_input = tokenizer.prepare_for_model(input_ids) # adds the special tokens [CLS] and [SEP]
# #                                                      # adds attention masks
# # print(final_input)
'''
impractical, just do everything in one stpe
'''
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/bert_tokenizer")
inputs = tokenizer("Let's try to tokenize!")
# print(inputs)
print(inputs["input_ids"])
# to access the special tokens such as [CLS] and [SEP]:
print(tokenizer.cls_token_id, tokenizer.sep_token_id)
# to decode the tokens:
print(tokenizer.decode(inputs["input_ids"]))