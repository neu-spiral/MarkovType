"""Train RNN models on the preprocessed data."""
import argparse
import sys

import numpy as np
from loguru import logger

from bci_disc_models.conf import (
    NAME_MODEL_RNN_KWARGS_LARGE,
    NAME_MODEL_RNN_KWARGS_MEDIUM,
    NAME_MODEL_RNN_KWARGS_SMALL,
)
from bci_disc_models.utils import PROJECT_ROOT, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lambda_loss", type=float, default=0.1)
    parser.add_argument("--tiny", action="store_true", default=False)
    parser.add_argument(
        "--reward", choices=["Linear", "Rational", "InverseSquare", "InverseCube", "InverseFourth"], default="Linear"
    )
    parser.add_argument("--model_size", type=str, default="medium")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    args.data_dir = r"preprocessed/"

    args.outdir = PROJECT_ROOT / "results"
    logger.add(args.outdir / "log.train.txt", level="DEBUG")
    logger.info(f"Running with args: {args}")

    with logger.catch(onerror=lambda _: sys.exit(1)):
        seed = args.seeds
        seed_everything(seed)
        logger.info(f"Load data for seed: {seed}...")
        # Load this split
        train_x = np.load(args.data_dir + f"train_x.seed_{seed}.npy", mmap_mode="r")
        train_y = np.load(args.data_dir + f"train_y.seed_{seed}.npy", mmap_mode="r")
        if args.tiny:
            logger.info("Tiny setup")
            # Get a few of each class
            train_class0_idx = np.where(train_y == 0)[0][:1000]
            train_class1_idx = np.where(train_y == 1)[0][:1000]
            idx = np.concatenate([train_class0_idx, train_class1_idx])
            train_x = train_x[idx]
            train_y = train_y[idx]
        logger.info("done loading")

        n_nontargets = np.sum(train_y == 0)
        n_targets = np.sum(train_y == 1)
        logger.info(f"Original ratio of nontarget/target is: {n_nontargets/n_targets}")
        if args.model_size == "large":
            NAME_MODEL_RNN_KWARGS = NAME_MODEL_RNN_KWARGS_LARGE
        elif args.model_size == "medium":
            NAME_MODEL_RNN_KWARGS = NAME_MODEL_RNN_KWARGS_MEDIUM
        elif args.model_size == "small":
            NAME_MODEL_RNN_KWARGS = NAME_MODEL_RNN_KWARGS_SMALL
        folder = (
            args.outdir
            / f"seed_{seed}"
            / args.model_size
            / f"epochs{args.epochs}"
            / f"lambdaloss{args.lambda_loss}"
            / args.reward
        )
        folder.mkdir(exist_ok=True, parents=True)
        for name, model_cls, model_kwargs in NAME_MODEL_RNN_KWARGS:
            logger.info(f"Try model: {model_cls.__name__}, {model_kwargs}")
            model_kwargs["lambda_loss"] = args.lambda_loss
            model_kwargs["model_size"] = args.model_size
            model_kwargs["epochs"] = args.epochs
            model_kwargs["reward"] = args.reward
            model = model_cls(**model_kwargs)
            model.fit_RNN(train_x, train_y)
            model.save(folder)
