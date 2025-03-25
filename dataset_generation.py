import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

import os

def generate_random_binary_structure(n):
    """Generate a random binary structure of size n x n."""
    return np.random.randint(0, 2, (n, n), dtype=np.uint8)*255

def apply_gaussian_filter(image, sigma):
    """Apply a Gaussian filter to an image."""
    return gaussian_filter(image, sigma=sigma)

def binarize_image(image, threshold=255/2):
    """Binarize an image based on a threshold."""
    return (image > threshold).astype(int)*255

def save_image_as_png(array, filename='image.png'):
    """Save the image as a PNG."""
    #image = Image.fromarray(array,"L")
    #image.save(filename)
    plt.imsave(filename, array, cmap='gray')

k = 2
n = 64

datasize = 30000


directory = 'datasets/'+str(n)+'/sigma'+str(k)+'/struct/'

import os
if not os.path.exists(directory):
    os.makedirs(directory)


for i in range(datasize):
    binary_structure = generate_random_binary_structure(n)
    noisy_structure = apply_gaussian_filter(binary_structure, k)
    binarized_structure = binarize_image(noisy_structure)
    print(binarized_structure)
    save_image_as_png(binarized_structure, filename='datasets/'+str(n)+'/sigma'+str(k)+'/struct/'+str(i)+'.png')



with Image.open('datasets/'+str(n)+'/sigma'+str(k)+'/struct/1.png') as img:
    width, height = img.size

    print(width, height)
    print(img.mode)