Original work: https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0

Pulled in changes from the forks to combine them into one and started rewriting it.

The goal was to have a StyleGAN2 implementation which can run on an Apple MacBook Pro M1 Pro/Max using the GPU.

Modifications from:
- https://github.com/StuartFarmer
- https://github.com/marmig0404
- https://github.com/jimmiebtlr
- https://github.com/Fofeu
- https://github.com/robgon-art
- https://github.com/leoHeidel

# StyleGAN2 Tensorflow 2.0

Unofficial implementation of StyleGAN 2 using TensorFlow 2.0. (Compatible with up to 2.5)

Original paper: Analyzing and Improving the Image Quality of StyleGAN

Arxiv: https://arxiv.org/abs/1912.04958


This implementation includes all improvements from StyleGAN to StyleGAN2, including:

Modulated/Demodulated Convolution, Skip block Generator, ResNet Discriminator, No Growth,

Lazy Regularization, Path Length Regularization, and can include larger networks (by adjusting the cha variable).



## Image Samples
Trained on Landscapes for 3.48 million images (290k steps, batch size 12, channel coefficient 24):
*To clarify, 3.48 million images were shown to the Discriminator, but the dataset consists of only ~14k images.
Thus, of those 3.48 million images, most are repeats of already seen images.*




## Before Running
Please ensure you have created the following folders:
1. /Models/
2. /Results/
3. /data/

Additionally, please ensure that your folder with images is in /data/ and changed at the top of stylegan.py.

