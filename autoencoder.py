import numpy as np
import matplotlib.pyplot as plt


class autoencoder:

    def __init__(self):
        self.H = 0  # Number of principle components

        self.principle_components = None
        self.sample_mean = None
        self.transformed_data = None
        self.decoded_data = None

        self.e_val = None
        self.e_vec = None

        self.type = None

        pass

    def pca_train(self, samples, p, h=0):
        """
        :param samples:
        :param p: Total Variance Explanied between 0 and 1
        :param h
        """

        self.type = 'PCA'

        self.sample_mean = samples.mean()  # Compute Sample mean
        centered_samples = samples - self.sample_mean  # Center samples around mean

        sample_cov = np.cov(centered_samples, rowvar=False)  # Find sample covariance matrix

        e_val, e_vec = sorted_eigen(sample_cov)  # Eigen Decomposition of sample covariance matrix
        # They have to be sorted in descending order

        if h == 0:
            sum = e_val[0]
            stop_val = p * np.trace(sample_cov)
            h = 0
            while sum < stop_val:
                h += 1
                sum += e_val[h]

        self.H = h

        print("The number of principle components is:", self.H)

        self.principle_components = e_vec[:, 0:self.H]  # First H components
        self.e_val = e_val
        self.e_vec = e_vec

    def gram_pca_train(self, samples, p, h=0):

        self.type = 'Gram'

        centering_matrix = np.identity(len(samples)) - (1/len(samples))*np.ones((len(samples), len(samples)))

        centered_sample_matrix = np.dot(centering_matrix, samples)
        self.gram_matrix = np.dot(centered_sample_matrix, centered_sample_matrix.T)

        double_centered_gram = centering_matrix @ self.gram_matrix @ centering_matrix

        e_val, e_vec = sorted_eigen(double_centered_gram)

        if h == 0:
            sum = e_val[0]
            stop_val = p * np.trace(self.gram_matrix)
            h = 0
            while sum < stop_val:
                h += 1
                sum += e_val[h]

        self.H = h

        print(self.H)

        self.principle_components = np.dot(np.dot(centering_matrix, e_vec[:, 0:self.H]), \
                                           np.linalg.inv(np.sqrt(np.diag(e_val[0:self.H]))))

        print(self.principle_components.shape)

        self.e_vec = e_vec
        self.e_val = e_val

    def encode(self, test_samples, training_samples=None):
        """

        :param training_samples:
        :param test_samples:
        :return:
        """

        if self.type == 'PCA':
            centered_samples = test_samples - self.sample_mean
            self.transformed_data = np.dot(centered_samples, self.principle_components)

        elif self.type == "Gram":
            test_vs_training_gram = np.dot(test_samples, training_samples.T)
            centered_tvt_gram = test_vs_training_gram - (1/len(training_samples))*self.gram_matrix
            self.transformed_data = np.dot(centered_tvt_gram, self.principle_components)

    def decode(self):
        """

        :return:
        """
        if self.type == 'PCA':
            self.decoded_data = np.dot(self.transformed_data, self.principle_components.T).astype(dtype=float) \
                                + self.sample_mean

        elif self.type == "Gram":
            print(self.transformed_data.shape)
            self.decoded_data = self.principle_components @ self.transformed_data.T


    def plot_cumulative_explained_variance(self):
        """
        Plots the cumulative explained variance vs the Number of components
        """
        plt.stem(np.cumsum(self.e_val / np.sum(self.e_val)))
        plt.axis('tight')
        plt.grid()
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.show()


def sorted_eigen(matrix):
    """
    Finds the Eigen values and vectors, and reorders them in descending order
    :param matrix: Matrix that will be EVD
    :return: Eigen value and Eigen vector in descending order
    """
    e_val, e_vec = np.linalg.eig(matrix)

    index = e_val.argsort()[::-1]
    e_val = e_val[index]
    e_vec = e_vec[index]

    return e_val, e_vec
