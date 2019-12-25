import tensorflow as tf
import numpy as np
import os

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

class dataGenerator(object):

    def __init__(self, folder, im_size, batch_size, flip = True, verbose = True):
        self.folder = folder
        self.im_size = im_size
        self.batch_size = batch_size
        self.flip = flip
        self.verbose = verbose

        self.segments = []
        self.images = []
        self.update = 0
        
        image_paths = tf.data.Dataset.list_files(folder + "*.jpg")
        
        if self.verbose:
            print(tf.data.experimental.cardinality(image_paths).numpy(), "images found")

        def im_preprocessing(im_path) :
            """
            Reading images from files, data augmentation should be here.
            """
            im_file = tf.io.read_file(im_path)
            im = tf.io.decode_jpeg(im_file, channels=3)
            im = tf.image.resize(im, (im_size,im_size))
            im = tf.image.convert_image_dtype(im, tf.float32)/255
            if flip :
                im = tf.image.random_flip_left_right(im)
                
            im = tf.image.resize(im, (im_size,im_size))
            return im

        dataset = image_paths.map(im_preprocessing)
        dataset = dataset.repeat()
        self.iterator = iter(dataset.batch(batch_size))
            
    def get_batch(self):
        self.update+=1
        if not self.update % 10 :
            print(self.update)
        return next(self.iterator)


