import h5py
from PIL import Image
import numpy as np
from pathlib import Path

# set the dimensions
pixel_dims = 128

# create a path for the images to be looped through
img_folder = Path("images")

data, target = [], []

for img_path in img_folder.iterdir():
    # load image and convert to RGB
    img = Image.open(img_path).resize((pixel_dims, pixel_dims)).convert("RGB")
    # convert to numpy array
    img_array = np.array(img)
    # add the image to the data
    data.append(img_array)

    # check if the image is a cat to add the target
    if "cat" in img_path.name:
        target.append(1)
    else:
        target.append(0)

# convert everything into numpy arrays
data, target = np.array(data), np.array(target)

# create the h5 database file
with h5py.File("data/database.h5", "w") as f:
    f.create_dataset("data", data=data, compression="gzip")
    f.create_dataset("target", data=target, compression="gzip")
