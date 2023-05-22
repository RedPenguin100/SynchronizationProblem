import time
import numpy as np

from math_utils import average
from measure_sync_library import array_validation


class OptAlgorithm:
    best_apriori = 'best_apriori'
    best_possible = 'best_possible'
    pure_random = 'pure_random'
    stupid_solution = 'stupid_solution'
    sync_mra = 'sync_mra'


class Setting:
    def __init__(self, sigma, samples, dimension, signal, algorithm):
        self.sigma = sigma
        self.samples = samples
        self.dimension = dimension

        self.signal = signal
        self.validate_signal()

        self.algorithm = algorithm

    def __str__(self):
        return f"{self.sigma},{self.samples},{self.dimension}"

    def validate_signal(self):
        array_validation(self.signal)
        assert self.signal.shape[0] == self.dimension

    def print_summary(self):
        print("Signal: ", self.signal)

        if self.sigma == 0.:
            print("Noiseless setting")
        else:
            print(f"SNR={1 / self.sigma}")

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

    def print(self, verbose=True):
        print(f"Wrong samples average: {average(self.wrong_samples)}")
        print(f"Average reconstruction error: {average(self.reconstruction_errors)}")
        print(f"Worst reconstruction error: {np.max(self.reconstruction_errors)}")
        print(f"Total results duration: {self.total_duration}")
        if verbose:
            print("Wrong samples list:", self.wrong_samples)
            print("Apriori best wrongs:", self.reconstruction_errors)

    def add_result(self, result: Result):
        self.wrong_samples.append(result.wrong_samples)
        self.reconstruction_errors.append(result.reconstruction_error)
        self.total_duration += result.duration
