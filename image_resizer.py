# image_resizer.py
# Importing required libraries
import os
import numpy as np
from PIL import Image

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = '/Users/Gomu/Downloads/Neural-Style-Transfer/d2'

images_path = IMAGE_DIR

training_data = []

count = 1
for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS).convert('RGB')
    training_data.append(np.asarray(image))
    print(f'{path}', count)
    count += 1

print('np reshaping...')
training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('saving file...')
np.save('art_data.npy', training_data)
