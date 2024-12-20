"""Print results table from parsed results and metrics."""
import argparse
import pickle as pkl

import numpy as np
from rich.console import Console
from rich.table import Table

from bci_disc_models.conf import METRICS
from bci_disc_models.utils import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_loss", type=float, default=0.1)
parser.add_argument("--rnn", action="store_true", default=False)
parser.add_argument("--model_size", type=str, default="medium")
parser.add_argument("--model_type", type=str, default="cnn-1d")
parser.add_argument(
    "--reward", choices=["Linear", "Rational", "InverseSquare", "InverseCube", "InverseFourth"], default="Linear"
)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--argmax_lastattempt", action="store_true", default=False)
args = parser.parse_args()
results_dir = PROJECT_ROOT / "results"

if args.rnn:

    name = (
        f"parsed_results_{args.model_size}_epochs_{args.epochs}_lambdaloss_{args.lambda_loss}_reward_{args.reward}.pkl"
    )

    if args.model_size == "large":
        models = ["large-rnn"]
    elif args.model_size == "medium":
        models = ["medium-rnn"]
    elif args.model_size == "small":
        models = ["small-rnn"]
else:
    name = f"parsed_results_{args.model_type}.pkl"
    if args.model_type == "cnn-1d":
        models = ["disc-nn-simple-cnn-1d"]
    elif args.model_type == "cnn-2d":
        models = ["disc-nn-simple-cnn-2d"]
if args.argmax_lastattempt:
    name = name.replace("parsed_results", "argmax_lastattempt_parsed_results")
with open(results_dir / name, "rb") as f:
    parsed_results = pkl.load(f)
METRICS.remove("test_acc")
METRICS.remove("test_bal_acc")
METRICS.remove("itr_typed")
METRICS.remove("acc_typed")
METRICS.remove("itr_argmax")
METRICS.remove("itr_query")
METRICS.remove("acc_argmax")
columns = ["Model"] + METRICS
rows = []
for model in models:
    row = []
    row.append(model)
    for metric in METRICS:

        mean = np.mean(parsed_results[model][metric])
        std = np.std(parsed_results[model][metric])
        row.append(f"{mean:.3f} ± {std:.3f}")
    rows.append(row)


table = Table(title="Model Comparison")
for col in columns:
    table.add_column(col, no_wrap=True)
for row in rows:
    table.add_row(*list(map(str, row)))
console = Console()
console.print(table)
