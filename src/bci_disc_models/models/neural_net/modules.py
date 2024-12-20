from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is adapted from the pytorch implementation[1] of the paper Recurrent models of visual attention[2].

    [1] www.github.com/kevinzakka/recurrent-visual-attention
    [2] Minh et. al., https://arxiv.org/abs/1406.6247
"""


class CombineNetwork(nn.Module):
    """Feature mapping.

    Map the given hidden state to possible classes using query letters.

    Args:
        G_t: a 2D tensor of shape (query_length, hidden_size).
        query_letters: a 1D tensor of shape (query_length).

    Returns:
        G_t: a 1D tensor of shape (n_classes*hidden_size).
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, features, query_letters):
        g_t = torch.zeros((28, features.shape[1])).to(self.device)
        g_t[query_letters.numpy(), :] = features
        return g_t.flatten()


class CoreNetwork(nn.Module):
    """The core network.

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        G_t: Alphabet feature vector for the current time step `t`.
        h_t_prev: Hidden state of the previous time step `t-1`.

    Returns:
        h_t: a 1D tensor of shape (n_classes * hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = self.ln(F.relu(h1 + h2))
        return h_t


class BaselineNetwork(nn.Module):
    """The baseline network.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (K, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t


class ClassifierNetwork(nn.Module):
    """The classification network.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        p_t: output probability vector over the classes.
    """

    def __init__(self, _hidden_size, n_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(_hidden_size, n_classes),
            nn.LogSoftmax(-1),
        )

    def forward(self, h_t):
        return self.classifier(h_t)


class FeatureNetwork_large(nn.Module):
    """The feature network.
    Get the feature of the input data by using 1D CNN for time t.
    output: G_t for letters presented
    """

    def __init__(self, n_dim: int, input_shape: Tuple[int]):
        super().__init__()
        self.net = self._CNN_N_DIM(n_dim, input_shape[0])

    def _CNN_N_DIM(self, n_dim=1, input_channels=1):
        if n_dim == 1:
            conv, bn = nn.Conv1d, nn.BatchNorm1d
        elif n_dim == 2:
            conv, bn = nn.Conv2d, nn.BatchNorm2ds
        else:
            raise ValueError("Only 1D and 2D CNNs are supported")
        return nn.Sequential(
            conv(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            bn(32),
            nn.SiLU(inplace=True),
            conv(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            bn(64),
            nn.SiLU(inplace=True),
            conv(64, 128, kernel_size=3, stride=2, bias=False),
            bn(128),
            nn.SiLU(inplace=True),
            conv(128, 64, kernel_size=3, stride=2, bias=False),
            bn(64),
            nn.SiLU(inplace=True),
            conv(64, 32, kernel_size=3, stride=2, bias=False),
            bn(32),
            nn.Flatten(),
        )

    def forward(self, data):
        return self.net(data)


class FeatureNetwork_medium(nn.Module):
    """The feature network.
    Get the feature of the input data by using 1D CNN for time t.
    output: g_t for letters presented
    """

    def __init__(self, n_dim: int, input_shape: Tuple[int]):
        super().__init__()
        self.net = self._CNN_N_DIM(n_dim, input_shape[0])

    def _CNN_N_DIM(self, n_dim=1, input_channels=1):
        if n_dim == 1:
            conv, bn = nn.Conv1d, nn.BatchNorm1d
        elif n_dim == 2:
            conv, bn = nn.Conv2d, nn.BatchNorm2ds
        else:
            raise ValueError("Only 1D and 2D CNNs are supported")
        return nn.Sequential(
            conv(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            bn(32),
            nn.SiLU(inplace=True),
            conv(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            bn(64),
            nn.SiLU(inplace=True),
            conv(64, 128, kernel_size=3, stride=2, bias=False),
            bn(128),
            nn.SiLU(inplace=True),
            conv(128, 48, kernel_size=3, stride=2, bias=False),
            bn(48),
            nn.SiLU(inplace=True),
            conv(48, 24, kernel_size=3, stride=2, bias=False),
            bn(24),
            nn.Flatten(),
        )

    def forward(self, data):
        return self.net(data)


class FeatureNetwork_small(nn.Module):
    """The feature network.
    Get the feature of the input data by using 1D CNN for time t.
    output: g_t for letters presented
    """

    def __init__(self, n_dim: int, input_shape: Tuple[int]):
        super().__init__()
        self.net = self._CNN_N_DIM(n_dim, input_shape[0])

    def _CNN_N_DIM(self, n_dim=1, input_channels=1):
        if n_dim == 1:
            conv, bn = nn.Conv1d, nn.BatchNorm1d
        elif n_dim == 2:
            conv, bn = nn.Conv2d, nn.BatchNorm2ds
        else:
            raise ValueError("Only 1D and 2D CNNs are supported")
        return nn.Sequential(
            conv(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            bn(32),
            nn.SiLU(inplace=True),
            conv(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            bn(64),
            nn.SiLU(inplace=True),
            conv(64, 128, kernel_size=3, stride=2, bias=False),
            bn(128),
            nn.SiLU(inplace=True),
            conv(128, 32, kernel_size=3, stride=2, bias=False),
            bn(32),
            nn.SiLU(inplace=True),
            conv(32, 16, kernel_size=3, stride=2, bias=False),
            bn(16),
            nn.Flatten(),
        )

    def forward(self, data):
        return self.net(data)
