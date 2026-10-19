from transformers import AutoConfig
from transformers import BertConfig, GPT2Config, BartConfig
from transformers import BertModel

'''
As seen previously, the automodel class allows us to instantiate a pretrained model from any checkpoint from the
hugging face library

it will pick the right model class to instantiate the weights and architecture of the pretrained model

behind the scene, it will cache the checkpoint's config file and model file (which contains the weights)
It is also possible to specify a local folder to hold the config and model weight file

to instantiate the model, automodel will first open the config file to look at the config class that should be used
the config class will be determined by the checkpoint.

After instantiating the right config class, it will find a model config that works
the config class also finds the correct model class, which uses the model config to load the correct model
Note that this model is not a pretrained model as it has random weights

the last step is to load the weights from the model file to the model. this will be the pretrained model
'''

'''
the AutoConfig API allows instantiation of the configuration of a pretrained model from any checkpoint
'''
bert_config = AutoConfig.from_pretrained("bert-base-cased")
# print(type(bert_config))

gpt_config = AutoConfig.from_pretrained("gpt2")
# print(type(gpt_config))

bart_config = AutoConfig.from_pretrained("facebook/bart-base")
# print(type(bart_config))

'''
auto config is a factory that figures out which specific config class to use and returns an instance of
that class

Specific configs like BartConfig is one of those specific classes. It defines the actual hyperparameters
'''

# bert_config = BertConfig.from_pretrained("bert-base-cased")
# print(type(bert_config))
#
# gpt_config = GPT2Config.from_pretrained("gpt2")
# print(type(gpt_config))
#
# bart_config = BartConfig.from_pretrained("facebook/bart-base")
# print(type(bart_config))

'''
These essentially works the same as the AutoConfig. The only difference is when to use them.

AutoConfig is for applications where the specific model is not known
the specific config is for applications where the specific model is known
'''

'''
these configurations contains all the information needed to load the model
'''
print(bert_config)
# these configurations can also be saved to a folder locally
bert_config.save_pretrained("./configs/bert_config")
# to load this next time, use the following code
# bert_config = BertConfig.from_pretrained("./configs/bert_config")

'''
Now the models can be modified through the config
'''
# this will produce the same as loading it from pretrained
bert_model = BertModel(bert_config)
# or edit the config while loading it
bert_config_custom = BertConfig.from_pretrained("bert-base-cased", num_hidden_layers=10)
bert_config_custom.save_pretrained("./configs/bert_config_custom")
bert_model_custom = BertModel(bert_config_custom)

'''
Saving is not exclusive to config; the models and tokenizers can also be saved
'''
bert_model_custom.save_pretrained("./models/bert_custom")
# and now it can be loaded like
bert_model_custom_loaded = BertModel.from_pretrained("./models/bert_custom")