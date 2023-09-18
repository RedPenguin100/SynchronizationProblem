import time
import numpy as np

from math_utils import average
from measure_sync_library import array_validation


class OptAlgorithm:
    measure_best_apriori = 'measure_best_apriori'
    measure_best_apriori_fourier = 'measure_best_apriori_fourier'
    best_possible = 'best_possible'
    pure_random = 'pure_random'
    stupid_solution = 'stupid_solution'
    sync_mra = 'sync_mra'


class Setting:
    def __init__(self, sigma, samples, dimension, signal, algorithm, outliers=0.):
        self.sigma = sigma
        self.samples = samples
        self.dimension = dimension

        self.signal = signal
        self.validate_signal()

        self.algorithm = algorithm
        self.outliers = outliers  # as a proportion, between 0 and 1

    def __str__(self):
        return f"{self.sigma},{self.samples},{self.dimension},{self.outliers}"

    def validate_signal(self):
        array_validation(self.signal)
        assert self.signal.shape[0] == self.dimension

    def print_summary(self):
        print("Signal: ", self.signal)

        if self.sigma == 0.:
            print("Noiseless setting")
        else:
            print(f"SNR={1 / self.sigma}")
        if self.outliers != 0.:
            print("Setting contain outliers")
        # TODO: calculate sigma adjusted for outliers

        print(f"Algorithm: {self.algorithm}")


class Result:
    def __init__(self, wrong_samples, reconstruction_error, duration, seed=None):
        self.wrong_samples = wrong_samples
        self.reconstruction_error = reconstruction_error
        self.duration = duration
        self.seed = seed


class Experiment:
    def __init__(self, setting):
        self.setting = setting
        self.wrong_samples = []
        self.reconstruction_errors = []
        self.timestamp = time.time()
        self.total_duration = 0.
        self.average_reconstruction_error = average(self.reconstruction_errors)

    def print(self, verbose=True):
        print(f"Wrong samples average: {average(self.wrong_samples)}")
        print(f"Average reconstruction error: {self.average_reconstruction_error}")
        print(f"Worst reconstruction error: {np.max(self.reconstruction_errors)}")
        print(f"Total results duration: {self.total_duration}")
        if verbose:
            print("Wrong samples list:", self.wrong_samples)
            print("Apriori best wrongs:", self.reconstruction_errors)

    def add_result(self, result: Result):
        self.wrong_samples.append(result.wrong_samples)
        self.reconstruction_errors.append(result.reconstruction_error)
        self.total_duration += result.duration
        self.average_reconstruction_error = average(self.reconstruction_errors)


class ComparisonMetric:
    wrong_samples = "wrong_samples"
    reconstruction_errors = "reconstruction_errors"
