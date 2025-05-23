import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.mps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHABET_LEN = 28
DECISION_THRESHOLD = 0.8
TRIALS_PER_SEQUENCE = 10
N_CHARS_TO_SPELL = 28
TOTAL_ATTEMPTS = 10
QUERY_SELECTION_METHOD = "sampleK"


def get_git_hash():
    """Get short git hash, with "+" suffix if local files modified"""
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")

    # Add '+' suffix if local files are modified
    exitcode, _ = subprocess.getstatusoutput("git diff-index --quiet HEAD")
    if exitcode != 0:
        h += "+"
    return "git" + h


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)


if __name__ == "__main__":
    h = get_git_hash()
    print("git hash: ", h)

    print("PROJECT_ROOT: ", PROJECT_ROOT)
