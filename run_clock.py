"""
Function to both load in data from a Beta CSV File
and run the clock returning a list of predictions for each
sample
"""

import csv
import numpy as np
import argparse


def read_file(beta_file, probes):
    """
    Function to read in beta file as CSV.
    Returns a tuple of:
        1. 2D Numpy array filtered for model probes
        2. List of samples
    """
    samples = None
    data = []
    probe_to_indx = {}
    for i, probe in enumerate(probes):
        data.append([])
        probe_to_indx[probe] = i
    with open(beta_file, newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
        for row in csv_reader:
            if not samples:
                samples = list(row.keys())[1:]
            if row[""] in probes:
                drow = []
                for sample in samples:
                    drow.append(float(row[sample]))
                data[probe_to_indx[row[""]]] = drow

    ## Missing probes
    missing_probes = []
    data_no_missing = []
    for i, k in enumerate(data):
        if len(k) == 0:
            missing_probes.append(i)
        else:
            data_no_missing.append(k)
    if len(missing_probes) > 0:
        print(f"WARNING: Missing probes (n={len(missing_probes)}) in beta file")
        print("continuing although predictions may be inaccurate")

    data = np.array(data_no_missing)
    newx = np.transpose(data)
    return (newx, samples, missing_probes)


def run_clock(beta_file):
    """
    Function to run clock given the path to a CSV
    file containing the normalised beta values
    """
    probes = np.load("model/probes.npy")
    newx, samples, missing_probes = read_file(beta_file, probes)
    nbeta = np.load("model/nbeta.npy")
    if len(missing_probes) > 0:
        nbeta = np.delete(nbeta, missing_probes, axis=0)
    result = np.dot(np.column_stack((np.ones([newx.shape[0], 1]), newx)), nbeta)[:, 0]
    return [[v, result[i]] for i, v in enumerate(samples)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    clock_results = run_clock(args.beta_file)
    with open(args.output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "CellPopAge"])
        writer.writerows(clock_results)
