import torch
import torch.nn as nn

'''
the following are the three methods that are used

# can use tensor, models, or dictionaries for saving
# uses python's pickle modules to serialize the model (human can't read it)
torch.save(arg, PATH)

torch.load(PATH)

model.load_state_dict(arg)

### COMPLETE MODEL ###
# only saves the parameters
torch.save(model.state_dict(), PATH)

# model must be created again with the parameters
model = Model(*args, **kwargs)
# load the weights back in
model.load_state_dict(torch.load(PATH))
model.eval()

models (and parameters) are conventionally saved in .pth files

state_dict is not exclusive to models. Any nn object has this, and it contains the parameters
and its value at that point

therefore you can save much more than just the model/model parameter

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": model.state_dict(),
}

torch.save(checkpoint, "checkpoint.pth")

# the loaded checkpoint behaves like how it was before it was saved
# so this will still be a dictionary
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

'''