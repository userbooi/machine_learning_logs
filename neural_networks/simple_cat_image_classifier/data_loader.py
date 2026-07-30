import h5py
import numpy as np
import pandas as pd

# create the data loader function
def data_loader():

    # load the dataset
    dataset = h5py.File("data/database.h5", "r")

    # return the data in the database split
    return dataset["data"][:], dataset["target"][:], dataset["list_class"][:]
