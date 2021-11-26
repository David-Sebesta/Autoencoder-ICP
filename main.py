from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import autoencoder
from random import sample
import idx2numpy


# TODO:
#   Fix PCA
#   Make getting samples better
#   Better display
#


if __name__ == "__main__":

    # Simple 8x8 digts
    # digits = load_digits()  # 8x8 digits
    # X = digits.data / 255
    # Y = np.array(digits.target, dtype=int)

    # 28x28 bit digits
    training_samples = idx2numpy.convert_from_file('samples/train-images.idx3-ubyte')
    Y = idx2numpy.convert_from_file('samples/train-labels.idx1-ubyte')

    training_samples = training_samples.reshape(len(training_samples), training_samples[0].size)
    training_samples = training_samples[:] / 255


    test_samples = idx2numpy.convert_from_file('samples/t10k-images.idx3-ubyte')
    test_y = idx2numpy.convert_from_file('samples/t10k-labels.idx1-ubyte')

    test_samples = test_samples.reshape(len(test_samples), test_samples[0].size)
    test_samples = test_samples[:] / 255


    auto = autoencoder.autoencoder()

    auto.pca_train(training_samples, 0.5)

    auto.encode(test_samples)

    auto.decode()

    for i in range(8):  # show 8 examples
        print("This is supposed to be a '", test_y[i], "':", sep="")
        plt.imshow(auto.decoded_data[i].reshape([28, 28]))
        plt.show()

