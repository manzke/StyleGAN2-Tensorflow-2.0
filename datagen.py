from PIL import Image
import numpy as np
import random
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


class DataGenerator(object):
    def __init__(self, path, dataset, img_size, mss = (1024 ** 3), flip = True, verbose = True):
        self.path = path
        self.dataset = dataset
        self.img_size = img_size
        self.segment_length = mss // (img_size * img_size * 3)
        self.flip = flip
        self.verbose = verbose

        self.segments = []
        self.images = []
        self.update = 0

        if self.verbose:
            print("Importing images from ...{}/{}".format(path, dataset))
            print("Maximum Segment Size: ", self.segment_length)

        numpy_path = self.path + "/" + self.dataset + "-npy-" + str(self.img_size)

        if os.path.exists(numpy_path) and len(os.listdir(numpy_path)) > 0:
            self.load_from_npy()
        else:
            os.makedirs(numpy_path, exist_ok = True)
            self.folder_to_npy()
            self.load_from_npy()

    def folder_to_npy(self):

        if self.verbose:
            print("Converting from images to numpy files...")

        names = []

        for dirpath, dirnames, filenames in os.walk(self.path + "/" + self.dataset):
            for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".JPEG"))]:
                fname = os.path.join(dirpath, filename)
                names.append(fname)

        np.random.shuffle(names)

        if self.verbose:
            print(str(len(names)) + " images.")

        kn = 0
        sn = 0

        segment = []

        for fname in names:
            if self.verbose:
                print('\r' + str(sn) + " // " + str(kn) + "\t", end = '\r')

            try:
                temp = Image.open(fname).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
            except:
                print("Importing image failed on", fname)
            temp = np.array(temp, dtype='uint8')
            segment.append(temp)
            kn = kn + 1

            if kn >= self.segment_length:
                np.save(self.path + "/" + self.dataset + "-npy-" + str(self.img_size) + "/data-"+str(sn)+".npy", np.array(segment))

                segment = []
                kn = 0
                sn = sn + 1


        np.save(self.path + "/" + self.dataset + "-npy-" + str(self.img_size) + "/data-"+str(sn)+".npy", np.array(segment))


    def load_from_npy(self):

        for dirpath, dirnames, filenames in os.walk(self.path + "/" + self.dataset + "-npy-" + str(self.img_size)):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                self.segments.append(os.path.join(dirpath, filename))

        self.load_segment()

    def load_segment(self):

        if self.verbose:
            print("Loading segment")

        segment_num = random.randint(0, len(self.segments) - 1)

        self.images = np.load(self.segments[segment_num])

        self.update = 0

    def get_batch(self, num):

        if self.update > self.images.shape[0]:
            self.load_from_npy()

        self.update = self.update + num

        idx = np.random.randint(0, self.images.shape[0] - 1, num)
        out = []

        for i in idx:
            out.append(self.images[i])
            if self.flip and random.random() < 0.5:
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32') / 255.0
