"""Parse saved results from model evaluation into a convenient form."""
import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from bci_disc_models.conf import (
    ALPHABET_LEN,
    DECISION_THRESHOLD,
    METRICS,
    TRIALS_PER_SEQUENCE,
)
from bci_disc_models.utils import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_loss", type=float, default=0.1)
parser.add_argument("--model_size", type=str, default="medium")
parser.add_argument("--model_type", type=str, default="cnn-1d")
parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results")
parser.add_argument("--rnn", action="store_true", default=False)
parser.add_argument("--argmax_lastattempt", action="store_true", default=False)
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
    return round(x, 4)


def compute_ITR_stats(typing_history, argmax_lastattempt=False):
    # Typing history contains lists with ragged shapes, such as: (attempted_letters, queries_per_letter, ALPHABET_LEN)

    # Extract parameters used during typing
    alphabet_len = typing_history.get("alphabet_len", ALPHABET_LEN)
    trials_per_sequence = typing_history.get("trials_per_sequence", TRIALS_PER_SEQUENCE)
    decision_threshold = typing_history.get("decision_threshold", DECISION_THRESHOLD)
    # target_letter_idx = typing_history.get("target_letter_idx", TARGET_LETTER_IDX)
    target_letter_idx = typing_history["true_letter_indices"]

    # Compute ITR in terms of bits per attempted letter using decision threshold, or using argmax
    all_log_posteriors = typing_history["pred_log_posteriors"]
    n_correct_decision_threshold = 0  # Compute accuracy according to DECISION_THRESHOLD
    n_correct_argmax = 0  # Compute accuracy using argmax on the final log_posteriors
    n_total = len(all_log_posteriors)
    mean_attempts = 0
    attempts_array = np.zeros(n_total)
    attempts_correctness = np.zeros(n_total)
    i = 0
    for log_posteriors in all_log_posteriors:
        mean_attempts += np.array(log_posteriors).size / ALPHABET_LEN
        attempts_array[i] = np.array(log_posteriors).size / ALPHABET_LEN
        if np.exp(log_posteriors[-1][target_letter_idx[i]]) >= decision_threshold:
            n_correct_decision_threshold += 1
            attempts_correctness[i] = 1

        if np.argmax(log_posteriors[-1]) == target_letter_idx[i]:
            n_correct_argmax += 1
            if (
                argmax_lastattempt
                and attempts_array[i] == 10
                and np.exp(log_posteriors[-1][target_letter_idx[i]]) < decision_threshold
            ):
                n_correct_decision_threshold += 1
                attempts_correctness[i] = 1
        i += 1
    mean_attempts /= n_total
    # Compute ITR of each query, using argmax amongst the K+1 probability buckets of each update
    n_correct_query = 0
    n_total_query = 0
    Z = zip(typing_history["pred_log_likelihoods"], typing_history["queried_letter_indices"])
    i = 0
    for several_log_likelihoods, several_queried_letter_idx in Z:
        # While trying to type this letter, we showed multiple queries
        for log_likelihoods, queried_letter_idx in zip(several_log_likelihoods, several_queried_letter_idx):
            # In each query, there are 11 "buckets" - the 10 letters shown, and everything else.
            # If we take the log likelihoods, group into these buckets, normalize, and take argmax,
            # we can ask whether the target letter's bucket was chosen. If so, we count it as a correct query.
            trials_per_sequence = len(queried_letter_idx)
            n_buckets = trials_per_sequence + 1
            buckets = np.zeros(n_buckets)
            buckets[:trials_per_sequence] = np.exp(log_likelihoods[queried_letter_idx])
            unseen_letter_idx = np.setdiff1d(np.arange(alphabet_len), queried_letter_idx)
            buckets[-1] = np.sum(np.exp(log_likelihoods[unseen_letter_idx]))
            # Determine which bucket is "correct".
            # If the target letter is in queried_letter_idx, that is the correct bucket.
            # Otherwise, buckets[-1] is the correct bucket.
            n_total_query += 1
            if target_letter_idx[i] in queried_letter_idx:
                correct_bucket_idx = np.where(queried_letter_idx == target_letter_idx[i])[0][0]
            else:
                correct_bucket_idx = n_buckets - 1
            if buckets.argmax() == correct_bucket_idx:
                n_correct_query += 1
        i += 1
    typed_letters = np.array(typing_history["pred_letter_indices"])
    n_typed = np.sum(typed_letters != None)
    n_correct_typed = np.sum(typed_letters == target_letter_idx)
    acc_typed = n_correct_typed / n_typed if n_typed > 0 else 0
    acc_decision_threshold = n_correct_decision_threshold / n_total
    acc_argmax = n_correct_argmax / n_total
    acc_query = n_correct_query / n_total_query

    return {
        "acc_typed": acc_typed,
        "acc_decision_threshold": acc_decision_threshold,
        "acc_argmax": acc_argmax,
        "acc_query": acc_query,
        "itr_typed": R(ITR(alphabet_len, acc_typed)),
        "itr_decision_threshold": R(ITR(alphabet_len, acc_decision_threshold)),
        "itr_decision_threshold_mean": R(ITR(alphabet_len, acc_decision_threshold) / mean_attempts),
        "itr_argmax": R(ITR(alphabet_len, acc_argmax)),
        "itr_query": R(ITR(alphabet_len, acc_query)),
        "n_typed": n_typed,
        "attempts": mean_attempts,
        "attempts_array": attempts_array,
        "attempts_correctness": attempts_correctness,
    }


results_dir = PROJECT_ROOT / "results"
if args.rnn:
    if args.model_size == "large":
        models = ["large-rnn"]
    elif args.model_size == "medium":
        models = ["medium-rnn"]
    elif args.model_size == "small":
        models = [f"small-rnn_{args.reward}"]
else:
    if args.model_type == "cnn-1d":
        models = ["disc-nn-simple-cnn-1d"]
    elif args.model_type == "cnn-2d":
        models = ["disc-nn-simple-cnn-2d"]
# For each model, we have several metrics to track.
# Each metric will have a list of 5 values (one for each seed).
all_results = {
    model: {metric: [] for metric in METRICS + ["attempts_array", "attempts_correctness"]} for model in models
}
attempts_over_seeds = []
attempts_correctness_over_seeds = []
for seed in range(5):
    if args.rnn:
        typing_history_files = list(
            (
                results_dir
                / f"seed_{seed}"
                / args.model_size
                / f"epochs{args.epochs}"
                / f"lambdaloss{args.lambda_loss}"
                / args.reward
            ).glob(f"typing_stats.{args.model_size}-rnn.pkl")
        )
    else:
        typing_history_files = list(
            (results_dir / f"seed_{seed}/{args.model_type}").glob(f"typing_stats.*{args.model_type}.pkl")
        )
    print(typing_history_files)
    for file in typing_history_files:
        if args.rnn:
            model = file.stem.replace("typing_stats.", "") + "_" + file.parent.stem
        else:
            model = file.stem.replace("typing_stats.", "")
        with open(file, "rb") as f:
            hist = pkl.load(f)
        res = {}
        res["n_params"] = hist["n_params"]
        itr_stats = compute_ITR_stats(hist, argmax_lastattempt=args.argmax_lastattempt)
        res.update(**itr_stats)
        for metric in METRICS + ["attempts_array", "attempts_correctness"]:
            if metric != "test_acc" and metric != "test_bal_acc":
                try:
                    all_results[model][metric].append(R(res[metric]))
                except KeyError:
                    breakpoint()
                except TypeError:
                    all_results[model][metric].append(res[metric])
        attempts_over_seeds = np.append(attempts_over_seeds, res["attempts_array"])
        attempts_correctness_over_seeds = np.append(attempts_correctness_over_seeds, res["attempts_correctness"])
# pprint(all_results)
if args.rnn:
    name = (
        f"parsed_results_{args.model_size}_epochs_{args.epochs}_lambdaloss_{args.lambda_loss}_reward_{args.reward}.pkl"
    )
    plot_name = (
        f"attempts_hist_{args.model_size}_epochs_{args.epochs}_lambdaloss_{args.lambda_loss}_reward_{args.reward}.pdf"
    )

else:
    plot_name = f"attempts_hist_{args.model_type}.pdf"
    name = f"parsed_results_{args.model_type}.pkl"
if args.argmax_lastattempt:
    name = name.replace("parsed_results", "argmax_lastattempt_parsed_results")
    plot_name = plot_name.replace("attempts_hist", "argmax_lastattempt_attempts_hist")
with open(results_dir / name, "wb") as f:
    pkl.dump(all_results, f)

# Plotting
if model == "small-rnn_Linear":
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
# Create arrays for counts of correct and incorrect attempts
max_attempts = 10
correct_counts = np.zeros(
    max_attempts,
)
incorrect_counts = np.zeros(
    max_attempts,
)

for i in range(1, max_attempts + 1):
    correct_counts[i - 1] = np.sum((attempts_over_seeds == i) & (attempts_correctness_over_seeds == 1))
    incorrect_counts[i - 1] = np.sum((attempts_over_seeds == i) & (attempts_correctness_over_seeds == 0))

# Set up the bar plot
x = np.arange(1, max_attempts + 1)
width = 0.7

fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(12, 12)
rects1 = ax.bar(x, correct_counts, width, label="Correct")
rects2 = ax.bar(x, incorrect_counts, width, bottom=correct_counts, label="Incorrect")
# Add grid lines
ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.9, alpha=0.9)
# Add some text for labels, title, and axes ticks
plt.title(f"Distribution of sequences", fontsize=38, fontweight="bold")
plt.xlabel("Number of sequences", fontsize=38, fontweight="bold")
plt.ylabel("Frequency", fontsize=38, fontweight="bold")
plt.xticks(x, fontsize=36)
plt.yticks(fontsize=36)
plt.legend(fontsize=36)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(attempts_over_seeds)))
plt.ylim(0, 1700)
fig.tight_layout()
plt.savefig(plot_name)
