This repository includes the code used in our work:

> Sunger E., Bicer, Y., Erdogmus, Imbiriba, T., "MarkovType: A Markov Decision Process Strategy for Non-invasive BCI Systems", Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

Please cite this paper ([preprint](https://arxiv.org/pdf/2412.15862)) if you intend to use this code for your research.

This work proposes a Markov Decision Process for non-invasive BCI typing systems (MarkovType) and formulate the
BCI typing procedure as a Partially Observable Markov Decision Process (POMDP), incorporating the typing
mechanism into the learning procedure. We compare the performance of MarkovType with previous approaches using Recursive Bayesian Estimation following https://ieeexplore.ieee.org/document/10095715. 

This repository was forked (then detached) from [bci-disc-models](https://github.com/nik-sm/bci-disc-models/), which is (c) 2022 Niklas Smedemark-Margulies and released under the MIT License.

We use https://pypi.org/project/thu-rsvp-dataset/1.1.0/ for fetching and preprocessing benchmark dataset from https://www.frontiersin.org/articles/10.3389/fnins.2020.568000/full.

# Setup

Setup project with `make` and activate virtualenv with `source venv/bin/activate`

# Usage

To reproduce our experiments, please follow these steps:

1. Preprocess data: `python scripts/prepare_data.py`
2. Pretrain baseline models: `python scripts/train.py`
3. Pretrain MarkovType models: `python scripts/train_rnn.py`
4. Evaluate models in simulated typing task: `python scripts/evaluate.py`
5. Parse saved results from evaluation with threshold: `python scripts/parse_results.py`
6. Parse saved results from evaluation without threshold: `python scripts/parse_results_without_threshold.py`
7. Collect statistics from parsed results: `python scripts/analyze_results.py`
8. Make plots: `python scripts/plot_metrics.py`
