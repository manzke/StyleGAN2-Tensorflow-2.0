"""
convert_to_jpeg.py directory
Used to convert a directory of images to jpg format
Martin Miglio (marmig0404) 2021
"""

import os
import sys

import PIL.Image as Image

source_dir = sys.argv[1]

for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(source_dir)):

    print(filenames)
    for file in filenames:
        infile = os.path.join(source_dir, file)
        f, e = os.path.splitext(infile)
        outfile = f + ".jpg"
        if infile != outfile:
            try:
                with Image.open(infile) as im:
                    im.save(outfile)
            except OSError:
                print("cannot convert", infile)
