from pprint import pprint

'''
Hugging face transformers are pretrained transformer models that performs NLP tasks

The pipeline function is the most high-level API of the transformers library
it combines all the steps to go from raw texts to usable predictions
    - includes the necessary preprocessing, model, and decoding outputs
    - one step to a usable prediction
It is the simplest way to create models
'''
from transformers import pipeline

# # pass in the function
# classifier = pipeline("sentiment-analysis")
# print(classifier("I've been waiting for the new GTA6.")) # <- perceives as negative as I am complaining the time
#                                                          #    I had to wait
#
# '''
# you can also place multiple sentences into one classifier to get outputs for all of them
# '''
# print(classifier([
#     # this sentence is still classified as negative, although the confidence score is much lower now
#     # since this could go both ways
#     "I've wanted the new GTA6 for my whole life.",
#     # the exclamation mark differentiates this sentence as a positive one
#     "I've wanted the new GTA6 for my whole life!",
#     # this is a clearly positive sentence
#     "I've always wanted this new GTA6 game"
# ]))

'''
The zero-shot prompting  is where you ask a model to do something without ever being trained on that 
specific task

Similar to one-shot learning for Face Recognition, however, one-shot learning gives the model one picture to
learn from, and zero-shot learning gives it zero

zero-shot learning is most realistic in NLP tasks like classification or translation

zero-shot classification however, is a general version of sentiment analysis, where it is capable of 
classifying a sentence beyond just positive or negative
'''
# classifier = pipeline("zero-shot-classification")
# prediction = classifier(
#     "This is a program containing notes and snippets about Hugging Face Transformers",
#     candidate_labels=["education", "politics", "business", "coding"]
# )
# print(prediction)

'''
the text generation pipeline will auto fill out the rest of the prompt
'''
# generator = pipeline("text-generation")
# # the outcome for the generation will be random as
# autofilled_text = generator([
#     "This program that uses a pre-trained MobileNetV2, will be",
#     "when I slice into a potato, the smell is"
# ])
# for gen_text in autofilled_text:
#     # print(gen_text)
#     print(gen_text[0]["generated_text"], end="\n\n")

'''
huggingface.co/models has models for different tasks.

up until now, we have only used the default model associated with each task. But any model that has been fine-tuned
or pre-trained on this task also works. Specify this in the model argument.
'''
# # for this example, use the gpt2-light for text generation
# generator = pipeline("text-generation", model="gpt2")
# predictions = generator(
#     "when I slice into a potato, the smell is",
#     max_new_tokens=250, # length of new tokens generated
#     num_return_sequences=2 # number of generations to return
# )
# print(predictions[0]["generated_text"], end='\n\n')
# print(predictions[1]["generated_text"])

'''
The BERT model uses only the Encoder portion of a transformer to do classification and prediction tasks within a 
given sentence.

the fill-mask pipeline uses a BERT model to fill in certain missing words, and chooses the ones with the highest
probability
'''
# unmasker = pipeline("fill-mask")
# # gives the top 2 most likely words for each mask
# words = unmasker("The Pythagorean Theorem, the most <mask> <mask> formula", top_k=2)
# pprint(words)

'''
NER function
'''
# ner = pipeline("ner")
# aggregation_strategy merges the sub tokens together, since modern transformers split input into not just words
ner = pipeline("ner", aggregation_strategy="simple")
recog = ner("My name is Edwin, and I am a student at UWaterloo")
pprint(recog)

'''
there are a few more models available
'''