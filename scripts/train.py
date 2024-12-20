"""Pretrain all models on each train split"""
import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from bci_disc_models.conf import NAME_MODEL_KWARGS_CNN_1D, NAME_MODEL_KWARGS_CNN_2D
from bci_disc_models.utils import PROJECT_ROOT, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tiny", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default="cnn-1d")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    args.data_dir = r"preprocessed/"

    args.outdir = PROJECT_ROOT / "results"
    if args.tiny:
        args.seeds = 2

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
            train_class0_idx = np.where(train_y == 0)[0][:10]
            train_class1_idx = np.where(train_y == 1)[0][:10]
            idx = np.concatenate([train_class0_idx, train_class1_idx])
            train_x = train_x[idx]
            train_y = train_y[idx]
        logger.info("done loading")

        n_nontargets = np.sum(train_y == 0)
        n_targets = np.sum(train_y == 1)
        logger.info(f"Original ratio of nontarget/target is: {n_nontargets/n_targets}")

        folder = args.outdir / f"seed_{seed}/{args.model_type}"
        folder.mkdir(exist_ok=True, parents=True)
        if args.model_type == "cnn-1d":
            NAME_MODEL_KWARGS = NAME_MODEL_KWARGS_CNN_1D
        elif args.model_type == "cnn-2d":
            NAME_MODEL_KWARGS = NAME_MODEL_KWARGS_CNN_2D
        for name, model_cls, model_kwargs in NAME_MODEL_KWARGS:
            logger.info(f"Try model: {model_cls.__name__}, {model_kwargs}")
            model = model_cls(**model_kwargs)
            model.fit(train_x, train_y)
            model.save(folder)
