import numpy as np


class autoencoder:

    def __init__(self):
        self.H = 0  # Number of principle components

        self.principle_components = None
        self.sample_mean = None
        self.transformed_data = None
        self.decoded_data = None

        pass

    def pca_train(self, samples, p, H=0):
        """
        :param samples:
        :param p: Total Variance Explanied between 0 and 1
        """

        self.sample_mean = samples.mean()                            # Compute Sample mean
        centered_samples = samples - self.sample_mean                # Center samples around mean

        sample_cov = np.cov(centered_samples, rowvar=False)     # Find sample covariance matrix

        e_val, e_vec = sorted_eigen(sample_cov)                 # Eigen Decomposition of sample covariance matrix
                                                                # They have to be sorted in decending order

        print(np.sum(e_val))

        if H == 0:
            sum = e_val[0]
            stop_val = p*np.trace(sample_cov)
            H = 0
            while sum < stop_val:
                H += 1
                sum += e_val[H]
            print(H+1)

        self.H = H

        self.principle_components = e_vec[:, 0:H]

    def encode(self, test_samples):

        centered_samples = test_samples - self.sample_mean

        self.transformed_data = np.dot(self.principle_components.T, centered_samples.T)

        pass

    def decode(self):

        self.decoded_data = np.dot(self.transformed_data.T, self.principle_components.T).astype(dtype=float) \
                            + self.sample_mean

        pass


def sorted_eigen(matrix):
    e_val, e_vec = np.linalg.eig(matrix)

    index = e_val.argsort()[::-1]
    e_val = e_val[index]
    e_vec = e_vec[index]

    return e_val, e_vec
