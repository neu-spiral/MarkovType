"Find best hyperparameters for a model using grid search."
import argparse
import pickle as pkl
from pathlib import Path

import numpy as np

from bci_disc_models.conf import ALPHABET_LEN, METRICS_plot
from bci_disc_models.utils import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_loss", type=list, default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
parser.add_argument("--model_size", type=str, default="small")
parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results")
parser.add_argument(
    "--reward", choices=["Linear", "Rational", "InverseSquare", "InverseCube", "InverseFourth"], default="Linear"
)
parser.add_argument("--epochs", type=int, default=200)
args = parser.parse_args()


def ITR(N, P):
    if P == 0:
        return 0
    if P == 1:
        return np.log2(N)
    return np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))


def R(x):
    return np.round(x, 4)


def compute_ITR_stats(typing_history):
    # Typing history contains lists with ragged shapes, such as: (attempted_letters, queries_per_letter, ALPHABET_LEN)

    # Extract parameters used during typing
    alphabet_len = typing_history.get("alphabet_len", ALPHABET_LEN)
    # target_letter_idx = typing_history.get("target_letter_idx", TARGET_LETTER_IDX)
    target_letter_idx = typing_history["true_letter_indices"]

    # Compute ITR in terms of bits per attempted letter using decision threshold, or using argmax
    all_log_posteriors = typing_history["pred_log_posteriors"]
    n_correct_argmax = np.zeros((10, 1))  # Compute accuracy using argmax on the final log_posteriors
    n_total = len(all_log_posteriors)

    for j in range(10):
        i = 0
        for log_posteriors in all_log_posteriors:
            try:
                if np.argmax(log_posteriors[j]) == target_letter_idx[i]:
                    n_correct_argmax[j] += 1
                i += 1
            except IndexError:
                breakpoint()
    acc_argmax = n_correct_argmax / n_total
    itr_argmax = np.zeros((10, 1))
    for j in range(10):
        itr_argmax[j] = R(ITR(alphabet_len, acc_argmax[j]) / (j + 1))
    return {
        "acc_argmax": acc_argmax,
        "itr_argmax": itr_argmax,
    }


results_dir = PROJECT_ROOT / "results"

data_dict = {}

if args.model_size == "large":
    models = ["large-rnn"]
elif args.model_size == "medium":
    models = ["medium-rnn"]
elif args.model_size == "small":
    models = [f"small-rnn_{args.reward}"]

# For each model, we have several metrics to track.
# Each metric will have a list of 5 values (one for each seed).

for lambda_loss in args.lambda_loss:
    all_results = {model + "_without_threshold": {metric: [] for metric in METRICS_plot} for model in models}
    for seed in range(5):
        typing_history_files = list(
            (
                results_dir
                / f"seed_{seed}"
                / args.model_size
                / f"epochs{args.epochs}"
                / f"lambdaloss{lambda_loss}"
                / args.reward
            ).glob("typing_stats_val.*_without_threshold.pkl")
        )
        print(typing_history_files)
        for model in models:
            model = model + "_without_threshold"
            for file in typing_history_files:
                with open(file, "rb") as f:
                    hist = pkl.load(f)
                res = {}
                itr_stats = compute_ITR_stats(hist)
                res.update(**itr_stats)
                for metric in METRICS_plot:
                    try:
                        all_results[model][metric].append(R(res[metric]))
                    except KeyError:
                        breakpoint()
    acc_last_attempts = np.array(all_results[model]["acc_argmax"])[:, 9, :]  # Get the last attempt accuracy
    data_dict[lambda_loss] = np.mean(acc_last_attempts)
# Print best accuracy and lambda
print(data_dict)
best_lambda = max(data_dict, key=data_dict.get)
print(f"Best lambda: {best_lambda}, best acc: {data_dict[best_lambda]}")
