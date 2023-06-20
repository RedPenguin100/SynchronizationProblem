import json
import os
from typing import List

from directories import DATA_DIR, DATA_SYNC_DIR, DATA_SYNC_DIR_OLD, DATA_DIR_OLD, DATA_STUPID_SOLUTION_DIR, \
    DATA_MEASURE_BEST_APRIORI, DATA_PURE_RANDOM, DATA_MEASURE_BEST_APRIORI_UNSQUARED, get_unconfirmed_data_directory
from math_utils import average
import matplotlib.pyplot as plt

from test_utils import ComparisonMetric


def get_data_files(data_dir) -> List:
    data_file_paths = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            data_file_paths.append(file_path)
    return data_file_paths


def get_json_list(data_dir):
    data_file_paths = get_data_files(data_dir)
    data_file_contents = []
    for file in data_file_paths:
        with open(file, 'r') as f:
            data_file_contents.append(json.load(f))
    return data_file_contents


def plot_by_error(json_list, dimension, samples, algo, color=None, linestyle=None, filter=None):
    if filter is None:
        raise Exception("Need to pass filter!")
    filtered_jsons = []
    for j in json_list:
        if j['setting']['samples'] != samples:
            continue
        if j['setting']['dimension'] != dimension:
            continue
        filtered_jsons.append(j)
    sigma_values = []
    average_error = []
    for filtered in filtered_jsons:
        sigma_values.append(filtered['setting']['sigma'])
        average_error.append(average(filtered[filter]))
    plt.plot(sigma_values, average_error, label=f"N={samples}, d={dimension}, alg={algo}", color=color,
             linestyle=linestyle)


def main():
    comparison_metric = ComparisonMetric.reconstruction_errors

    experiment_names = ['pure_random', 'measure_best_apriori', 'sync_mra', 'stupid_solution']
    json_list = [get_json_list(get_unconfirmed_data_directory(name)) for name in experiment_names]
    linestyles = ['dashed', 'solid', 'dotted', 'dashdot']
    for dimension, color in [
        (15, "black"),
        # (10, "green"),
        # (15, "blue")
    ]:
        for samples in [
            # 15,
            # 25,
            # 45,
            # 70,
            100
        ]:
            for experiment_data, algo, linestyle in zip(json_list, experiment_names, linestyles):
                plot_by_error(experiment_data, dimension=dimension, samples=samples, algo=algo,
                              filter=comparison_metric)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
