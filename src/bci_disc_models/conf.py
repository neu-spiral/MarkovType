from bci_disc_models.models import TrialNeuralNet

MODELS = [
    "disc-nn-simple-cnn-1d",
]
METRICS = [
    "n_params",
    "test_acc",
    "test_bal_acc",
    # NOTE - "itr_decision_threshold" is the notion of ITR we focus on, since
    # it corresponds most directly to actual typing performance
    "itr_decision_threshold",
    "itr_decision_threshold_mean",
    "itr_argmax",
    "itr_query",
    "itr_typed",
    "n_typed",
    "acc_typed",
    "acc_decision_threshold",
    "acc_argmax",
    "attempts",
]
METRICS_plot = [
    "itr_argmax",
    "acc_argmax",
]

TARGET_LETTER_IDX = 0  # TODO - this is hardcoded in 2 places (see src/evaluation.py)
ALPHABET_LEN = 28
DECISION_THRESHOLD = 0.8
TRIALS_PER_SEQUENCE = 10

# TODO - ideally this should be tunable when the model is being used during
# simulated typing. Right now, it is stored on the model so it is needed when model
# is created during training as well.
# Likewise - TRIALS_PER_SEQUENCE should be tunable during typing.
PRIOR_P_TARGET_IN_QUERY = TRIALS_PER_SEQUENCE / ALPHABET_LEN

# Add discriminative models
NAME_MODEL_KWARGS_CNN_1D = []
for arch in ["simple-cnn-1d"]:
    NAME_MODEL_KWARGS_CNN_1D.append(
        (
            f"disc-nn-{arch}",
            TrialNeuralNet,
            dict(
                prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY,
                arch=arch,
                input_shape=(62, 63),
                n_classes=2,
                epochs=25,
                lr=0.001,
            ),
        )
    )
NAME_MODEL_KWARGS_CNN_2D = []
for arch in ["simple-cnn-2d"]:
    NAME_MODEL_KWARGS_CNN_2D.append(
        (
            f"disc-nn-{arch}",
            TrialNeuralNet,
            dict(
                prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY,
                arch=arch,
                input_shape=(62, 63),
                n_classes=2,
                epochs=25,
                lr=0.001,
            ),
        )
    )
NAME_MODEL_RNN_KWARGS_LARGE = []
for arch in ["large-rnn"]:
    NAME_MODEL_RNN_KWARGS_LARGE.append(
        (
            f"{arch}",
            TrialNeuralNet,
            dict(
                prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY,
                arch=arch,
                input_shape=(62, 63),
                n_classes=28,
                epochs=25,
                lr=0.001,
            ),
        )
    )
NAME_MODEL_RNN_KWARGS_MEDIUM = []
for arch in ["medium-rnn"]:
    NAME_MODEL_RNN_KWARGS_MEDIUM.append(
        (
            f"{arch}",
            TrialNeuralNet,
            dict(
                prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY,
                arch=arch,
                input_shape=(62, 63),
                n_classes=28,
                epochs=25,
                lr=0.001,
            ),
        )
    )
NAME_MODEL_RNN_KWARGS_SMALL = []
for arch in ["small-rnn"]:
    NAME_MODEL_RNN_KWARGS_SMALL.append(
        (
            f"{arch}",
            TrialNeuralNet,
            dict(
                prior_p_target_in_query=PRIOR_P_TARGET_IN_QUERY,
                arch=arch,
                input_shape=(62, 63),
                n_classes=28,
                epochs=25,
                lr=0.001,
            ),
        )
    )
