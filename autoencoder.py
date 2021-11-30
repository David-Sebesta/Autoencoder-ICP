"""
Autoencoder ICP
David Sebesta
Pattern Recognition
Fall 2021
"""
import numpy as np
import matplotlib.pyplot as plt


class autoencoder:
    """
    An autoencoder using PCA
    """

    def __init__(self):
        """
        Creates a basic (linear) autoencoder using PCA.
        """
        self.H = 0  # Number of principle components

        self.principle_components = None  # Numpy array of principle component
        self.sample_mean = None  # Sample mean
        self.transformed_data = None  # Encoded or transformed data
        self.decoded_data = None  # Decoded Data

        self.e_val = None  # Eigen Values
        self.e_vec = None  # Eigen Vectors

        self.type = None  # PCA (Covariance Based) or Gram

    def pca_train(self, samples, p, h=None):
        """
        Finds the primary components using the covariance method. \n

        .. math::
            \\hat{\mu}_{x} = \\frac{1}{N}X^{T}1_{N}

            \\tilde{X} = X - 1_{N}\\hat{\mu}_{x}

            \\hat{C}_{x} = \\frac{1}{N}\\tilde{X}^{T}\\tilde{X}

            [\\nu , \\lambda] = EVD(\\hat{C}_{x})

            H = arg\\min\{k\\in\{1,...,D\}:\\sum_{i=1}^{k} \ \\lambda_{i} \\geq ptrace\{\\hat{C}_{x}\}\}

        :param samples: Training Samples
        :param p: Total Variance Explanied between 0 and 1
        :param h: Number of principle components wanted to be used. If no input, it is calculated based of p
        """

        self.type = 'PCA'  # Set the type of Autoencoder to PCA

        self.sample_mean = samples.mean()  # Compute Sample mean
        centered_samples = samples - self.sample_mean  # Center samples around mean

        sample_cov = np.cov(centered_samples, rowvar=False)  # Find sample covariance matrix

        e_val, e_vec = sorted_eigen(sample_cov)  # Eigen Decomposition of sample covariance matrix
        # They have to be sorted in descending order

        if h == None:
            sum = e_val[0]
            stop_val = p * np.trace(sample_cov)  # Total Variance explained * trace of the sample covariance matrix
            h = 0
            while sum < stop_val:               # When the sum of eigen values is greater than
                h += 1                          # the Total Variance explained * trace of the sample cov matrix
                sum += e_val[h]                  # That is when the number of principle components is found

        self.H = h  # Set the number of primary components
        print("The number of principle components is:", self.H + 1)

        self.principle_components = e_vec[:, 0:self.H]  # First H components
        self.e_val = e_val
        self.e_vec = e_vec

    def gram_pca_train(self, samples, p, h=0):
        """
        Finds the primary components using the gram matrix method.

        :param samples: Training Samples
        :param p: Total Variance Explanied between 0 and 1
        :param h: Number of principle components wanted to be used. If no input, it is calculated based of p
        """

        self.type = 'Gram'  # Set the type of autoencoder to Gram

        centering_matrix = np.identity(len(samples)) - (1 / len(samples)) * np.ones((len(samples), len(samples)))

        centered_sample_matrix = centering_matrix @ samples
        self.gram_matrix = np.dot(centered_sample_matrix, centered_sample_matrix.T)

        double_centered_gram = centering_matrix @ self.gram_matrix @ centering_matrix

        e_val, e_vec = sorted_eigen(double_centered_gram)

        if h == None:
            sum = e_val[0]
            stop_val = p * np.trace(self.gram_matrix)   # Total Variance explained * trace of the gram matrix
            h = 0
            while sum < stop_val:                       # When the sum of eigen values is greater than
                h += 1                                  # the Total Variance explained * trace of the gram matrix
                sum += e_val[h]                         # That is when the number of principle components is found

        self.H = h

        print("The number of principle components is:", self.H + 1)

        self.principle_components = centering_matrix @ e_vec[:, 0:self.H] @ np.linalg.inv(
            np.sqrt(np.diag(e_val[0:self.H])))

        self.e_vec = e_vec
        self.e_val = e_val

    def encode(self, test_samples, training_samples=None):
        """
        Encodes the test samples use the primary components. If Covariance style PCA is used, training samples does not
        need to be added. \n

        Covariance PCA:

        .. math::
            \\tilde{X}_{t} = X_t - 1_{M}\\hat{\\mu}^{T}_{x}

            \\tilde{Y}_{t} = \\tilde{X}_{t}\\nu_{H}

        If gram style PCA is used, training samples needs to be inputed otherwise the test samples will be used. \n


        :param test_samples: The samples that will be encoded
        :param training_samples: The samples used for training. Only needed from Gram PCA.
        """

        if self.type == 'PCA':
            centered_samples = test_samples - self.sample_mean.T
            self.transformed_data = test_samples @ self.principle_components  # Since the test samples are row vectors
                                                                              # no transpose is needed

        elif self.type == "Gram":
            if training_samples is None:
                training_samples = test_samples
            test_vs_training_gram = test_samples @ training_samples.T
            centered_tvt_gram = test_vs_training_gram - (1 / len(training_samples)) * self.gram_matrix
            self.transformed_data = centered_tvt_gram @ self.principle_components

    def decode(self):
        """
        Decodes the transformed data. \n
        Covariance PCA:

        .. math::
            \\hat{X}_{t} = \\tilde{Y}_{t}\\nu^{T}_{H} + \\hat{\\mu}_{x}


        Gram PCA:
            Unknown


        """
        if self.type == 'PCA':
            self.decoded_data = np.dot(self.transformed_data, self.principle_components.T).astype(dtype=float) \
                                + self.sample_mean.T

        elif self.type == "Gram":
            self.decoded_data = self.principle_components @ self.transformed_data.T

    def plot_cve(self):
        """
        Plots the cumulative variance explained vs the Number of components
        """
        plt.stem(np.cumsum(self.e_val / np.sum(self.e_val)))
        plt.axis('tight')
        plt.grid()
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.show()


def sorted_eigen(matrix):
    """
    Finds the Eigen values and vectors, and reorders them in descending order.

    :param matrix: Matrix that will be EVD
    :return: Eigen value and Eigen vector in descending order
    """
    e_val, e_vec = np.linalg.eig(matrix)

    index = e_val.argsort()[::-1]
    e_val = e_val[index]
    e_vec = e_vec[:, index]

    return e_val, e_vec
