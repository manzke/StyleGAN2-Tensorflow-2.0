import tensorflow as tf
import numpy as np
import os
import random

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
        print()

class DataGenerator(object):

    def __init__(self, image_paths, img_size, batch_size, flip = True, verbose = True):
        self.im_size = img_size
        self.batch_size = batch_size
        self.flip = flip
        self.verbose = verbose
        self.segments = []
        self.images = []
        self.update = 0
            
        def im_preprocessing(im_path) :
            """
            Reading images from files, data augmentation should be here.
            """
            im_file = tf.io.read_file(im_path)
            im = tf.io.decode_jpeg(im_file, channels=3)
            im = tf.image.resize(im, (img_size,img_size))
            im = tf.image.convert_image_dtype(im, tf.float32)/255
            
            im = tf.image.random_flip_left_right(im)
            im = tf.image.random_hue(im, 0.1)
            im = tf.image.rot90(im, random.randint(0, 3))
            
            random_top = random.randint(0, img_size // 20)
            random_left = random.randint(0, img_size // 20)
            random_bottom = img_size - random.randint(0, img_size // 20)
            random_right = img_size - random.randint(0, img_size // 20)
            new_height = random_bottom-random_top
            new_width = random_right-random_left
            im = tf.image.crop_to_bounding_box(im, random_top, random_left, new_height, new_width)
                
            im = tf.image.resize(im, (img_size,img_size))
            return im

        dataset = image_paths.map(im_preprocessing)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.iterator = iter(dataset)
            
    def get_batch(self):
        self.update+=1
        return next(self.iterator)
    