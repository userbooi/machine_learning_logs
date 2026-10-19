from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

'''
The datasets library is an API that allows one to download public datasets and preproces them
'''

'''
download the dataset

The first argument is the path:
if the dataset comes from the HF hub, use the full Hub dataset ID ("namespace/name") such as:
    - nyu-mll/glue, stanfordnlp/imdb, rajpurkar/squad
    - short names like "glue" no longer work in newer versions of huggingface_hub
if it comes from a local file, use the file format
    - csv, json, parquet, text
    
The second argument is the configuration. This is like a subset for a specific task
'''
raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
print(raw_datasets)
print()
# these datasets are indexed like a dictionary
print(raw_datasets["train"][0])

'''
There are also more arguments to control data loading further:

The third argument is the split:
    - loads only what is specified such as: "train", "test", of slices like 
      "train[:10%]" or "train[:500]"
    - this will return only a single Dataset, not DatasetDict
        - indexing will be different, not more dictionary indexing
        
the fourth argument is data_files:
    - path to my own files for the dataset
    - Ex. {"train": "train.csv", "test": "test.csv"}
    
the fifth argument is data_dir:
    - a folder of data files to load from
'''
# the splits can also take place after loading the entire dataset
validation_split = raw_datasets["validation"]
print()
print(validation_split)
print()
print(validation_split[4])
# the features of the datasets can also be viewed
# this gives specifications on the type and stuff that each column holds
print()
print(validation_split.features)

'''
to preprocess all elements of the dataset, it must be tokenized.

.map is used to apply a function to every element. This was also how to modify datasets in
pytorch or tensorflow
'''

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], padding=True, truncation=True, max_length=128
    )

# the batched=True is simply to speed up tokenizing
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# print(tokenized_datasets["train"][0])
# print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))

'''
remove the unusable columns from the dataset, and it is ready for training
'''
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "Labels")
tokenized_datasets = tokenized_datasets.with_format("torch")

train_loader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)
val_loader = DataLoader(tokenized_datasets["validation"], shuffle=True, batch_size=8)
test_loader = DataLoader(tokenized_datasets["test"], shuffle=True, batch_size=8)


