import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import tensorflow as tf

DATASET_PATH = 'data'
TRAIN_PATH = os.path.join(DATASET_PATH, 'fashiongen_256_256_validation.h5')

N_TRAIN_IMGS = 260480
BATCH_SIZE = 16

def h5_gen():
    with h5py.File(TRAIN_PATH, 'r') as hf:
        for im in hf["input_image"]:
            yield im

def train_preprocess(image):
    image /= 255

    #Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    return image

dataset = tf.data.Dataset.from_generator(
     h5_gen,
     (tf.int64),
     (tf.TensorShape([256,256,3]))
)

# Shuffle, repeat, batch
# dataset = dataset.shuffle(N_TRAIN_IMGS)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(1)

from stylegan_two_refactored import StyleGAN

model = StyleGAN(dataset, lr = 0.0001, silent = False)
model.evaluate(0)

while model.GAN.steps < 1001:
    model.train()

