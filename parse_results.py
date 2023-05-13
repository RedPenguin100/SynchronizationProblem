import json
import os
from typing import List

from directories import DATA_DIR, DATA_SYNC_DIR
from math_utils import average
import matplotlib.pyplot as plt


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


def plot_by_error(json_list, dimension, samples, algo, color, linestyle=None, filter=None):
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
    plt.plot(sigma_values, average_error, label=f"N={samples}, d={dimension}, alg={algo}", color=color, linestyle=linestyle)


def main():
    json_list_measure = get_json_list(DATA_DIR)
    json_list_sync = get_json_list(DATA_SYNC_DIR)
    for result_measure, result_sync in zip(json_list_measure, json_list_sync):
        wrong_samples = result_measure['reconstruction_errors']
        print("Measure Average wrong samples: ", average(wrong_samples))

        wrong_samples_sync = result_sync['reconstruction_errors']
        print("Sync Average wrong samples: ", average(wrong_samples_sync))

    for dimension, color in [(5, "red"), (10, "green"), (15, "purple")]:
        for samples in [
            # (15, "brown"),
            # (25, "purple"),
            45,
            # 70,
            # 100
        ]:
            plot_by_error(json_list_measure, dimension=dimension, samples=samples, algo='measure', color=color, linestyle='--', filter="reconstruction_errors")
            plot_by_error(json_list_sync, dimension=dimension, samples=samples, algo='sync', color=color, filter="reconstruction_errors")
    plt.legend(loc='upper left')
    plt.show()

def plot_by_samples():
    json_list_measure = get_json_list(DATA_DIR)
    json_list_sync = get_json_list(DATA_SYNC_DIR)
    for result_measure, result_sync in zip(json_list_measure, json_list_sync):
        wrong_samples = result_measure['wrong_samples']
        print("Measure Average wrong samples: ", average(wrong_samples))

        wrong_samples_sync = result_sync['wrong_samples']
        print("Sync Average wrong samples: ", average(wrong_samples_sync))

    for dimension, color in [(5, "red"), (10, "brown"), (15, "purple")]:
        for samples in [
            # (15, "brown"),
            # (25, "purple"),
            # (45, "black"),
            # (70, "blue"),
            100
        ]:
            plot_by_error(json_list_measure, dimension=dimension, samples=samples, algo='measure', color=None, linestyle='--')
            plot_by_error(json_list_sync, dimension=dimension, samples=samples, algo='sync', color=None)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()