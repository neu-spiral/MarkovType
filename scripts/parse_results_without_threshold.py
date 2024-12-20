"""Parse saved results from model evaluation into a convenient form."""
import argparse
import pickle as pkl
from pathlib import Path

import numpy as np

from bci_disc_models.conf import ALPHABET_LEN, METRICS_plot
from bci_disc_models.utils import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_loss", type=float, default=0.1)
parser.add_argument("--model_size", type=str, default="medium")
parser.add_argument("--model_type", type=str, default="cnn-1d")
parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results")
parser.add_argument("--rnn", action="store_true", default=False)
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
all_results = {model + "_without_threshold": {metric: [] for metric in METRICS_plot} for model in models}
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
            ).glob("typing_stats.*_without_threshold.pkl")
        )
    else:
        typing_history_files = list(
            (results_dir / f"seed_{seed}/{args.model_type}").glob("typing_stats.*_without_threshold.pkl")
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
# pprint(all_results)
acc_last_attempts = np.array(all_results[model]["acc_argmax"])[:, 9, :]
if args.rnn:
    print(
        f"Model {args.model_size}, reward {args.reward}, lambda_loss {args.lambda_loss}, Accuracy {np.mean(acc_last_attempts):.4f} ± {np.std(acc_last_attempts):.4f}"
    )
    name = f"parsed_results_without_threshold_{args.model_size}_epochs_{args.epochs}_lambdaloss_{args.lambda_loss}_reward_{args.reward}.pkl"

else:
    print(f"Model {args.model_type},  Accuracy {np.mean(acc_last_attempts):.4f} ± {np.std(acc_last_attempts):.4f}")
    name = f"parsed_results_without_threshold_{args.model_type}.pkl"
with open(args.outdir / name, "wb") as f:
    pkl.dump(all_results, f)
