import h5py
from PIL import Image
import numpy as np
from pathlib import Path

# set the dimensions
pixel_dims = 128

# create a path for the images to be looped through
img_folder = Path("images")

data, target, list_class = [], [], []

for img_path in img_folder.iterdir():
    # load image and convert to RGB
    raw_image = Image.open(img_path).resize((pixel_dims, pixel_dims))
    img = raw_image.convert("RGB")
    img_flipped1 = raw_image.transpose(method=Image.FLIP_LEFT_RIGHT).convert("RGB")
    img_flipped2 = raw_image.transpose(method=Image.FLIP_TOP_BOTTOM).convert("RGB")
    # convert to numpy array
    img_array = np.array(img)
    img_flipped_array1 = np.array(img_flipped1)
    img_flipped_array2 = np.array(img_flipped2)
    # add the image to the data
    data.append(img_array)
    data.append(img_flipped_array1)
    data.append(img_flipped_array2)

    # check if the image is a cat to add the target
    if "cat" in img_path.name:
        target.append(1); target.append(1); target.append(1)
        list_class.append("cat"); list_class.append("cat"); list_class.append("cat")
    else:
        target.append(0); target.append(0); target.append(0)
        list_class.append("not cat"); list_class.append("not cat"); list_class.append("not cat")

# convert everything into numpy arrays
data, target = np.array(data), np.array(target)

# create the h5 database file
with h5py.File("data/database.h5", "w") as f:
    f.create_dataset("data", data=data, compression="gzip")
    f.create_dataset("target", data=target, compression="gzip")
    f.create_dataset("list_class", data=list_class, compression="gzip")
