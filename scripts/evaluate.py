"""Load pretrained models and collect evaluation metrics on test splits."""
import argparse
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from bci_disc_models.conf import (
    ALPHABET_LEN,
    DECISION_THRESHOLD,
    NAME_MODEL_KWARGS_CNN_1D,
    NAME_MODEL_KWARGS_CNN_2D,
    NAME_MODEL_RNN_KWARGS_LARGE,
    NAME_MODEL_RNN_KWARGS_MEDIUM,
    NAME_MODEL_RNN_KWARGS_SMALL,
    TRIALS_PER_SEQUENCE,
)
from bci_disc_models.evaluation import simulate_typing, simulate_typing_RNN
from bci_disc_models.models import (
    LDA,
    AlwaysClass0,
    AlwaysClass1,
    GenerativeBaseline,
    LogR,
    ScikitModelAdaptor,
    TrialNeuralNet,
)
from bci_disc_models.utils import PROJECT_ROOT, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_loss", type=float, default=0.1)
parser.add_argument("--model_size", type=str, default="medium")
parser.add_argument("--model_type", type=str, default="cnn-1d")
parser.add_argument("--tiny", action="store_true", default=False)
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--n_chars_to_spell", type=int, default=1000)
parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results")
parser.add_argument("--rnn", action="store_true", default=False)
parser.add_argument(
    "--reward", choices=["Linear", "Rational", "InverseSquare", "InverseCube", "InverseFourth"], default="Linear"
)
parser.add_argument("--validation", action="store_true", default=False)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--without-threshold", action="store_true", default=False)
args = parser.parse_args()
args.data_dir = r"preprocessed/"
if args.tiny:
    args.seeds = 2
    args.n_chars_to_spell = 10

logger.add(args.outdir / "log.txt", level="DEBUG")
logger.add(args.outdir / "log.colorize.txt", level="DEBUG", colorize=True)
logger.info(f"Running with args: {args}")


def count_params(model):
    if isinstance(model, TrialNeuralNet):
        n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    elif isinstance(model, ScikitModelAdaptor):
        if isinstance(model.clf, (LogR, LDA)):
            n_params = model.clf.coef_.size + model.clf.intercept_.size
        else:
            raise NotImplementedError()
    elif isinstance(model, GenerativeBaseline):
        n_params = 0
        scaler = model.pipeline[0].model
        n_params += scaler.scale_.size + scaler.mean_.size
        pca = model.pipeline[2]
        n_params += pca.components_.size
        clf = model.pipeline[3].clf
        if isinstance(clf, (LogR, LDA)):
            n_params = clf.coef_.size + clf.intercept_.size
        else:
            raise NotImplementedError()
        kdes = model.pipeline[4].models
        n_params += kdes[0].tree_.data.size + kdes[1].tree_.data.size
    elif isinstance(model, (AlwaysClass0, AlwaysClass1)):
        n_params = 0
    else:
        raise NotImplementedError()
    return n_params


def evaluate_one_model(model, test_x, test_y, rnn_use=True, model_size=None, without_threshold=False):
    model_evaluation_dict = {}

    # Count params
    model_evaluation_dict["n_params"] = count_params(model)
    if not rnn_use:
        # Compute accuracy and save predicted probs
        probs = model.predict_proba(test_x)
        preds = probs.argmax(-1)
        test_acc = accuracy_score(test_y, preds)
        test_bal_acc = balanced_accuracy_score(test_y, preds)
        logger.info("Test Accuracy: {:.3f}, Balanced accuracy: {:.3f}".format(test_acc, test_bal_acc))
        model_evaluation_dict["probs"] = probs
        model_evaluation_dict["labels"] = test_y
        model_evaluation_dict["test_acc"] = test_acc
        model_evaluation_dict["test_bal_acc"] = test_bal_acc

    # Simulated typing metrics
    typing_params = dict(
        trials_per_sequence=TRIALS_PER_SEQUENCE,
        n_chars_to_spell=args.n_chars_to_spell,
        alphabet_len=ALPHABET_LEN,
        query_selection_method="sampleK",
        decision_threshold=DECISION_THRESHOLD,
    )

    if rnn_use:
        typing_results = simulate_typing_RNN(
            model=model,
            test_data=test_x,
            test_labels=test_y,
            model_size=model_size,
            without_threshold=without_threshold,
            **typing_params,
        )
    else:
        typing_results = simulate_typing(
            model=model, test_data=test_x, test_labels=test_y, without_threshold=without_threshold, **typing_params
        )
    model_evaluation_dict.update(**typing_results)
    model_evaluation_dict.update(**typing_params)

    return model_evaluation_dict


with logger.catch(onerror=lambda _: sys.exit(1)):
    for seed in range(args.seeds):

        logger.info(f"Load data for seed: {seed}...")
        # Load this split
        if args.validation:
            seed_everything(0)
            x = np.load(args.data_dir + f"train_x.seed_{seed}.npy", mmap_mode="r")
            y = np.load(args.data_dir + f"train_y.seed_{seed}.npy", mmap_mode="r")
            _, test_x, _, test_y = train_test_split(x, y, test_size=0.1)
        else:
            test_x = np.load(args.data_dir + f"test_x.seed_{seed}.npy", mmap_mode="r")
            test_y = np.load(args.data_dir + f"test_y.seed_{seed}.npy", mmap_mode="r")
        if args.tiny:
            # Get a few of each class
            test_class0_idx = np.where(test_y == 0)[0][:10]
            test_class1_idx = np.where(test_y == 1)[0][:10]
            idx = np.concatenate([test_class0_idx, test_class1_idx])
            test_x = test_x[idx]
            test_y = test_y[idx]
        logger.info("done loading")
        if args.rnn:
            folder = (
                args.outdir
                / f"seed_{seed}"
                / args.model_size
                / f"epochs{args.epochs}"
                / f"lambdaloss{args.lambda_loss}"
                / args.reward
            )
        else:
            folder = args.outdir / f"seed_{seed}/{args.model_type}"
        folder.mkdir(exist_ok=True, parents=True)
        if args.rnn:
            if args.model_size == "large":
                parameters = NAME_MODEL_RNN_KWARGS_LARGE
            elif args.model_size == "medium":
                parameters = NAME_MODEL_RNN_KWARGS_MEDIUM
            elif args.model_size == "small":
                parameters = NAME_MODEL_RNN_KWARGS_SMALL
        else:
            if args.model_type == "cnn-1d":
                parameters = NAME_MODEL_KWARGS_CNN_1D
            elif args.model_type == "cnn-2d":
                parameters = NAME_MODEL_KWARGS_CNN_2D
        seed_everything(seed)
        for name, model_cls, model_kwargs in parameters:
            logger.info(f"Try model: {model_cls.__name__}, {model_kwargs}")
            model = model_cls(**model_kwargs)
            model.load(folder)
            model_evaluation_dict = evaluate_one_model(
                model,
                test_x=test_x,
                test_y=test_y,
                rnn_use=args.rnn,
                model_size=args.model_size,
                without_threshold=args.without_threshold,
            )
            model_evaluation_dict["model_cls"] = model_cls.__name__
            model_evaluation_dict["model_kwargs"] = model_kwargs
            if args.without_threshold:
                name = f"{name}_without_threshold"
            if args.validation:
                with open(folder / f"typing_stats_val.{name}.pkl", "wb") as f:
                    pkl.dump(model_evaluation_dict, f)
            else:
                with open(folder / f"typing_stats.{name}.pkl", "wb") as f:
                    pkl.dump(model_evaluation_dict, f)
