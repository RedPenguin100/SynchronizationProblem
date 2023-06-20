import os

from test_utils import OptAlgorithm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR_OLD = os.path.join(PROJECT_ROOT, "data_measure_old")
DATA_MEASURE_BEST_APRIORI = os.path.join(PROJECT_ROOT, "data_measure_best_apriori")
DATA_MEASURE_BEST_APRIORI_UNSQUARED = os.path.join(PROJECT_ROOT, "data_measure_best_apriori_unsquared")
DATA_SYNC_DIR_OLD = os.path.join(PROJECT_ROOT, "data_sync_old")
DATA_PURE_RANDOM = os.path.join(PROJECT_ROOT, "data_pure_random")

DATA_DIR = os.path.join(PROJECT_ROOT, "data_measure")
DATA_STUPID_SOLUTION_DIR = os.path.join(PROJECT_ROOT, "data_stupid_solution")
DATA_SYNC_DIR = os.path.join(PROJECT_ROOT, "data_sync")


def get_unconfirmed_data_directory(experiment_name):
    return os.path.join(PROJECT_ROOT, "data_" + experiment_name)


def get_data_directory(algorithm: OptAlgorithm):
    if algorithm == OptAlgorithm.pure_random:
        return DATA_PURE_RANDOM
    if algorithm == OptAlgorithm.measure_best_apriori:
        return DATA_MEASURE_BEST_APRIORI
    if algorithm == OptAlgorithm.stupid_solution:
        return DATA_STUPID_SOLUTION_DIR
    raise ValueError("Sorry")
