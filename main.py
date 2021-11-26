from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
import autoencoder
from random import sample
import idx2numpy
from PIL import Image
from pathlib import Path


# TODO:
#   Fix PCA
#   Make getting samples better
#   Better display
#   Add Faces


def load_8x8_digits():
    """
    Loads 8 8x8 digits
    :return: Training Samples, Training Labels, Test Samples, Test Labels
    """
    digits = load_digits()  # 8x8 digits
    training_samples = digits.data / 255
    training_y = np.array(digits.target, dtype=int)
    test_samples = training_samples.copy()
    test_y = training_y.copy()
    image_shape = (8, 8)

    return training_samples, training_y, test_samples, test_y, image_shape


def load_28x28_digits():
    """
    Loads 60000 28x28 digits for training.
    Loads 10000 28x28 digits for testing.
    :return: Training Samples, Training Labels, Test Samples, Test Labels, Image Shape
    """
    # 28x28 bit digits
    training_samples = idx2numpy.convert_from_file('samples/train-images.idx3-ubyte')
    training_y = idx2numpy.convert_from_file('samples/train-labels.idx1-ubyte')

    image_shape = training_samples[0].shape

    training_samples = training_samples.reshape(len(training_samples), training_samples[0].size)  # Samples in one vector
    training_samples = training_samples[:] / 255  # Normalize between 0 and 1

    test_samples = idx2numpy.convert_from_file('samples/t10k-images.idx3-ubyte')
    test_y = idx2numpy.convert_from_file('samples/t10k-labels.idx1-ubyte')

    test_samples = test_samples.reshape(len(test_samples), test_samples[0].size)
    test_samples = test_samples[:] / 255

    return training_samples, training_y, test_samples, test_y, image_shape


def load_faces():

    if Path('samples/faces/face_data.npy').is_file():
        samples = np.load('samples/faces/face_data.npy', allow_pickle=True)
    else:
        filelist = []
        path = 'samples/faces/subject'
        for i in range(1,16):
            mod = ''
            if i < 10:
                mod = '0'
            filelist.append(path + mod + str(i) + '.centerlight')
            filelist.append(path + mod + str(i) + '.glasses')
            filelist.append(path + mod + str(i) + '.happy')
            filelist.append(path + mod + str(i) + '.leftlight')
            filelist.append(path + mod + str(i) + '.noglasses')
            filelist.append(path + mod + str(i) + '.normal')
            filelist.append(path + mod + str(i) + '.rightlight')
            filelist.append(path + mod + str(i) + '.sad')
            filelist.append(path + mod + str(i) + '.sleepy')
            filelist.append(path + mod + str(i) + '.surprised')
            filelist.append(path + mod + str(i) + '.wink')


        samples = np.array(([np.array(Image.open(fname)) for fname in filelist]))

        samples.dump('samples/faces/face_data.npy')  # Save samples so don't have to read each time

    samples = samples[:, 0:samples[0].shape[0], 0:samples[0].shape[1]]  # downsample by 4

    image_shape = samples[0].shape

    samples = samples.reshape(len(samples), samples[0].size)

    return samples, image_shape


if __name__ == "__main__":

    # training_samples, training_y, test_samples, test_y, image_shape = load_8x8_digits()

    samples, image_shape = load_faces()

    # plt.imshow(samples[0].reshape(image_shape), cmap='gray')
    # plt.show()

    # Create Auto encoder
    auto = autoencoder.autoencoder()

    auto.gram_pca_train(samples, 0.9)
    auto.encode(samples, samples)
    auto.decode()
    #
    # f = plt.figure()
    #
    # for i in range(2):  # show 8 examples
    #     plt.imshow(samples[i].reshape(image_shape), cmap='gray')
    #     plt.show()
    #     plt.imshow(auto.decoded_data[i].reshape(image_shape), cmap='gray')
    #     plt.show()

    auto.plot_cumulative_explained_variance()



