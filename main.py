"""
Autoencoder ICP
David Sebesta
Pattern Recognition
Fall 2021
"""

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import autoencoder
from random import sample
import idx2numpy
from PIL import Image
from pathlib import Path


def load_8x8_digits():
    """
    Uses the mnist 8x8 digits
    Loads 8 8x8 digits
    :return:
        Training Samples: NxD matrix of training samples
        Training Labels: N length vector of samples labels
        Test Samples: MxD matrix of testing samples
        Test Labels: M length vector of training labels
        Image Shape: Shape of image (8, 8)
    """

    digits = load_digits()                          # 8x8 digits from sklearn dataset
    training_samples = digits.data / 255            # Scale data
    training_y = np.array(digits.target, dtype=int) # Get training labels
    test_samples = training_samples.copy()          # Test samples are the same the training ## will be changed ##
    test_y = training_y.copy()                      # Get test labels
    image_shape = (8, 8)                            # Image shape is 8x8

    return training_samples, training_y, test_samples, test_y, image_shape


def load_28x28_digits():
    """
    Uses the mnist 28x28 digits.
    Loads 60000 28x28 digits for training.
    Loads 10000 28x28 digits for testing.
    :return:
        Training Samples: NxD matrix of training samples
        Training Labels: N length vector of samples labels
        Test Samples: MxD matrix of testing samples
        Test Labels: M length vector of training labels
        Image Shape: Shape of image (28, 28)
    """

    training_samples = idx2numpy.convert_from_file('samples/mnist_28x28_digits/train-images.idx3-ubyte')   # Load mnist training samples
    training_y = idx2numpy.convert_from_file('samples/mnist_28x28_digits/train-labels.idx1-ubyte')         # Load mnist training labels

    image_shape = training_samples[0].shape  # Finds the shape of the images

    training_samples = training_samples.reshape(len(training_samples), training_samples[0].size)  # Reshapes to NxD
    training_samples = training_samples[:] / 255  # Normalize between 0 and 1

    test_samples = idx2numpy.convert_from_file('samples/mnist_28x28_digits/t10k-images.idx3-ubyte')        # Load mnist testing samples
    test_y = idx2numpy.convert_from_file('samples/mnist_28x28_digits/t10k-labels.idx1-ubyte')              # Load mnist testing labels

    test_samples = test_samples.reshape(len(test_samples), test_samples[0].size)                # Reshapes to MxD
    test_samples = test_samples[:] / 255    # Normalizes between 0 and 1

    return training_samples, training_y, test_samples, test_y, image_shape


def load_cropped_faces(downsample=4):
    """
    Loads cropped yale faces. If file doesnt exists, it creates one.
    :param downsample: Amount the images will be downsampled
    :return:
        samples: The images in an N x D matrix, where N is the number of images and D is the number of pixels
        images_shape: The shape of the image to be used to reshape the matricies to plot
    """
    if Path('samples/cropped_faces/face_data.npy').is_file():
        samples = np.load('samples/cropped_faces/cropped_face_data.npy', allow_pickle=True)
    else:
        filelist = []

        for i in range(1, 166):
            filelist.append('samples/cropped_faces/sample' + str(i) + '.gif')
            samples = np.array(([np.array(Image.open(fname)) for fname in filelist]))

        samples = samples[:,:,:,0]
        samples.dump('samples/cropped_faces/copped_face_data.npy')  # Save samples so don't have to read each time

    samples = samples[:, 0:samples[0].shape[0]:downsample, 0:samples[0].shape[1]:downsample]  # downsample by 8

    image_shape = samples[0].shape

    samples = samples.reshape(len(samples), samples[0].size)
    samples = samples[:] / 255
    return samples, image_shape


def add_noise(samples, strength=0.5):
    """
    Add a uniform noise to the signal.

    :param samples: Samples
    :param strength: Max amplitude of noise
    :return:
        Samples added with a uniform noise from 0 to 1
    """
    noise = np.random.uniform(0, 1, samples[0].size)
    return samples[:] + noise * strength


def train_8x8_digits(p=0.9, h=None, noise_strength=0, n_images=2, plot_cve=False):
    """
    Built in autoencoder training and image showing. Uses mnist 8x8 digits. Change noise strength to a value between 0
    and 1 to add noise to the test samples.

    :param p: Total Variance Explained to be used
    :param h: Number of primary components wanted to be used. If left blank or None, it is found using p.
    :param noise_strength: Strength of the noise added to the test signal. Use a value between 0 and 1. Recommended
    between 0.001 and 0.1 anything greater will not be recognizable
    :param n_images: Amount of images to be shown
    :param plot_cev: If true, plots the Cumulative Explained Variance vs Number of principle components
    """

    training_samples, training_y, test_samples, test_y, image_shape = load_8x8_digits()

    auto = autoencoder.autoencoder()
    auto.pca_train(samples=training_samples, p=p, h=h)

    if noise_strength != 0:
        test_samples = add_noise(test_samples, strength=noise_strength)

    auto.encode(test_samples=test_samples)
    auto.decode()

    random_indexes = sample(range(0, len(test_samples)), n_images)

    for i in random_indexes:
        plt.subplot(1, 2, 1)
        plt.title('Test Sample: ' + str(test_y[i]))
        plt.imshow(test_samples[i].reshape(image_shape), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Decoded Sample: ' + str(test_y[i]))
        plt.imshow(auto.decoded_data[i].reshape(image_shape), cmap='gray')
        plt.show()

    if plot_cve:
        auto.plot_cve()


def train_28x28_digits(p=0.9, h=None, noise_strength=0, n_images=2, plot_cve=False):
    """
    Built in autoencoder training and image showing. Uses mnist 28x28 digits. Change noise strength to a value between 0
    and 1 to add noise to the test samples.

    :param p: Total Variance Explained to be used
    :param h: Number of primary components wanted to be used. If left blank or None, it is found using p.
    :param noise_strength: Strength of the noise added to the test signal. Use a value between 0 and 1.
    :param n_images: Amount of images to be shown
    :param plot_cev: If true, plots the Cumulative Explained Variance vs Number of principle components
    """
    training_samples, training_y, test_samples, test_y, image_shape = load_28x28_digits()

    auto = autoencoder.autoencoder()
    auto.pca_train(samples=training_samples, p=p, h=h)

    if noise_strength != 0:
        test_samples = add_noise(test_samples, strength=noise_strength)

    auto.encode(test_samples=test_samples)
    auto.decode()

    random_indexes = sample(range(0, len(test_samples)), n_images)

    for i in random_indexes:
        plt.subplot(1, 2, 1)
        plt.title('Test Sample: ' + str(test_y[i]))
        plt.imshow(test_samples[i].reshape(image_shape), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Decoded Sample: ' + str(test_y[i]))
        plt.imshow(auto.decoded_data[i].reshape(image_shape), cmap='gray')
        plt.show()

    if plot_cve:
        auto.plot_cve()


def train_faces(p=0.9, h=None, noise_strength=0, n_images=2, plot_cve=False, downsample=4):
    """
    Built in autoencoder and image showing. PCA trains an autoencoder using the cropped and downscaled yale faces.
    Change noise strength to a value between 0 and 1 to add noise to the test samples.

    :param p: Total Variance Explained to be used
    :param h: Number of primary components wanted to be used. If left blank or None, it is found using p.
    :param noise_strength: Strength of the noise added to the test signal. Use a value between 0 and 1.
    :param n_images: Amount of images to be shown
    :param plot_cev: If true, plots the Cumulative Explained Variance vs Number of principle components
    :param downsample: Amount the images will be downsampled (needs to be at least 3, not enough ram)
    """

    samples, image_shape = load_cropped_faces(downsample=downsample)

    # Create Auto encoder
    auto = autoencoder.autoencoder()

    auto.pca_train(samples=samples, p=p, h=h)

    print(samples[0].size)

    if noise_strength != 0:
        samples = add_noise(samples, strength=noise_strength)

    auto.encode(test_samples=samples)
    auto.decode()

    random_indexes = sample(range(0, len(samples)), n_images)

    for i in random_indexes:
        plt.subplot(1, 2, 1)
        plt.title('Test Sample')
        plt.imshow(samples[i].reshape(image_shape), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Decoded Sample')
        plt.imshow(auto.decoded_data[i].reshape(image_shape), cmap='gray')
        plt.show()

    if plot_cve:
        auto.plot_cve()


if __name__ == "__main__":

    train_8x8_digits(p=0.85, plot_cve=True)





