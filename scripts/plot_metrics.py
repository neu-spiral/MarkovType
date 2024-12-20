"""Print results table from parsed results and metrics."""
import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

from bci_disc_models.conf import METRICS_plot
from bci_disc_models.utils import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--with_threshold", action="store_true", default=False)
parser.add_argument("--seeds", type=int, default=5)
args = parser.parse_args()
results_dir = PROJECT_ROOT / "results"
out_dir = PROJECT_ROOT

data_dict = {}
if args.with_threshold:
    files = list((results_dir).glob("argmax_lastattempt_parsed_results*.pkl"))
else:
    files = list((results_dir).glob("parsed_results_without_threshold*.pkl"))
# Load each pickle file and update the dictionary
for file in files:
    with open(file, "rb") as f:
        data = pkl.load(f)
        data_dict.update(data)
if args.with_threshold:
    metric = "attempts_correctness"
    attempts = "attempts_array"
    plt.figure(figsize=(16, 14))
    markers = ["o", "s", "D", "^", "*", "H"]
    i = 0
    for model in data_dict.keys():
        attempts_correctness_over_seeds = np.array(data_dict[model][metric])
        attempts_over_seeds = np.array(data_dict[model][attempts])
        max_attempts = 10
        correct_counts = np.zeros([args.seeds, max_attempts])
        incorrect_counts = np.zeros([args.seeds, max_attempts])
        for j in range(args.seeds):
            for k in range(1, max_attempts + 1):
                correct_counts[j, k - 1] = np.sum(
                    (attempts_over_seeds[j] == k) & (attempts_correctness_over_seeds[j] == 1)
                )
                incorrect_counts[j, k - 1] = np.sum(
                    (attempts_over_seeds[j] == k) & (attempts_correctness_over_seeds[j] == 0)
                )
        accuracy = correct_counts / (correct_counts + incorrect_counts)
        means = np.mean(accuracy, axis=0)
        stds = np.std(accuracy, axis=0)
        x = np.arange(1, max_attempts + 1)
        if model == f"small-rnn_Linear":
            model_name = r"MarkovType $\left(\frac{2N-n-1}{N}\right)$"
        elif model == "small-rnn_Rational":
            model_name = r"MarkovType $\left(\frac{1}{n}\right)$"
        elif model == "small-rnn_InverseSquare":
            model_name = r"MarkovType $\left(\frac{1}{n^2}\right)$"
        elif model == "small-rnn_InverseCube":
            model_name = r"MarkovType $\left(\frac{1}{n^3}\right)$"
        elif model == "small-rnn_InverseFourth":
            model_name = r"MarkovType $\left(\frac{1}{n^4}\right)$"
        elif model == "disc-nn-simple-cnn-1d":
            model_name = "RB - 1D CNN"
        elif model == "disc-nn-simple-cnn-2d":
            model_name = "RB - 2D CNN"
        plt.plot(x, means, label=f"{model_name}", marker=markers[i], markersize=18, linewidth=4)
        plt.fill_between(x, means - stds, means + stds, alpha=0.2)
        i += 1

    # Adding labels and title
    plt.xlabel("Number of sequences", fontsize=28, fontweight="bold")
    metric_name = "Accuracy"
    plt.ylabel(f"{metric_name}", fontsize=28, fontweight="bold")
    # Get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Desired order
    desired_order = [
        "RB - 1D CNN",
        "RB - 2D CNN",
        r"MarkovType $\left(\frac{2N-n-1}{N}\right)$",
        r"MarkovType $\left(\frac{1}{n}\right)$",
        r"MarkovType $\left(\frac{1}{n^2}\right)$",
        r"MarkovType $\left(\frac{1}{n^3}\right)$",
    ]

    # Create a mapping from label to handle
    label_to_handle = dict(zip(labels, handles))

    # Reorder handles and labels based on the desired order
    ordered_handles = [label_to_handle[label] for label in desired_order]
    ordered_labels = [label for label in desired_order]

    # Create legend with sorted labels
    plt.legend(ordered_handles, ordered_labels, fontsize=24, loc="lower right")
    plt.xticks(x, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0, 1)
    # Add grid lines
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.9, alpha=0.9)
    plt.title(f"{metric_name} across number of sequences with threshold", fontsize=28, fontweight="bold")
    plt.savefig(f"{metric}_threshold.pdf", bbox_inches="tight")

else:
    # Plotting
    for metric in METRICS_plot:
        plt.figure(figsize=(16, 14))
        markers = ["o", "s", "D", "^", "*", "H"]
        i = 0
        for model in data_dict.keys():
            means = np.mean(data_dict[model][metric], axis=0)
            stds = np.std(data_dict[model][metric], axis=0)
            # Ensure that means and stds are 1D arrays
            means = np.array(means).flatten()
            stds = np.array(stds).flatten()

            x = np.arange(1, len(means) + 1)
            if model == f"small-rnn_Linear_without_threshold":
                model_name = r"MarkovType $\left(\frac{2N-n-1}{N}\right)$"
            elif model == "small-rnn_Rational_without_threshold":
                model_name = r"MarkovType $\left(\frac{1}{n}\right)$"
            elif model == "small-rnn_InverseSquare_without_threshold":
                model_name = r"MarkovType $\left(\frac{1}{n^2}\right)$"
            elif model == "small-rnn_InverseCube_without_threshold":
                model_name = r"MarkovType $\left(\frac{1}{n^3}\right)$"
            elif model == "small-rnn_InverseFourth_without_threshold":
                model_name = r"MarkovType $\left(\frac{1}{n^4}\right)$"
            elif model == "disc-nn-simple-cnn-1d_without_threshold":
                model_name = "RB - 1D CNN"
            elif model == "disc-nn-simple-cnn-2d_without_threshold":
                model_name = "RB - 2D CNN"
            plt.plot(x, means, label=f"{model_name}", marker=markers[i], markersize=18, linewidth=4)
            plt.fill_between(x, means - stds, means + stds, alpha=0.2)
            i += 1

        # Adding labels and title
        plt.xlabel("Number of sequences", fontsize=28, fontweight="bold")
        if metric == "itr_argmax":
            metric_name = "ITR (bits/sequence)"
        elif metric == "acc_argmax":
            metric_name = "Accuracy"
        plt.ylabel(f"{metric_name}", fontsize=28, fontweight="bold")
        # Get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Desired order
        desired_order = [
            "RB - 1D CNN",
            "RB - 2D CNN",
            r"MarkovType $\left(\frac{2N-n-1}{N}\right)$",
            r"MarkovType $\left(\frac{1}{n}\right)$",
            r"MarkovType $\left(\frac{1}{n^2}\right)$",
            r"MarkovType $\left(\frac{1}{n^3}\right)$",
        ]

        # Create a mapping from label to handle
        label_to_handle = dict(zip(labels, handles))

        # Reorder handles and labels based on the desired order
        ordered_handles = [label_to_handle[label] for label in desired_order]
        ordered_labels = [label for label in desired_order]

        # Create legend with sorted labels
        plt.legend(ordered_handles, ordered_labels, fontsize=24, loc="lower right")

        plt.xticks(x, fontsize=28)
        plt.yticks(fontsize=28)
        plt.ylim(0, 1)
        # Add grid lines
        plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.9, alpha=0.9)
        plt.title(f"{metric_name} across number of sequences without threshold", fontsize=28, fontweight="bold")
        plt.savefig(f"{metric}.pdf", bbox_inches="tight")
