from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm, trange

from bci_disc_models.evaluation import get_query
from bci_disc_models.utils import (
    ALPHABET_LEN,
    DECISION_THRESHOLD,
    N_CHARS_TO_SPELL,
    QUERY_SELECTION_METHOD,
    TOTAL_ATTEMPTS,
    TRIALS_PER_SEQUENCE,
)

from .dataloaders import Datamodule


class Trainer:
    def __init__(self, model, datamodule: Datamodule, lr: float, results_dir: Path, device: torch.device, tqdm_pos=0):
        self.device = device
        self.tqdm_pos = tqdm_pos

        # data
        self.n_classes = datamodule.n_classes
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = None if datamodule.val_set is None else datamodule.val_dataloader()

        # model
        self.model = model.to(device)
        self.optim = AdamW(self.model.parameters(), lr=lr)
        self.sched = ExponentialLR(self.optim, gamma=0.97)

        # loss functions
        class_weights = datamodule.class_weights.to(self.device)
        self.criterion = lambda log_probs, labels: F.nll_loss(log_probs, labels, weight=class_weights)

        # bookkeeping
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.results_dir)
        self.best_val_bal_acc = -torch.inf
        self.acc_metric = Accuracy(task="binary").to(self.device)
        self.bal_acc_metric = Accuracy(task="binary", num_classes=self.n_classes, average="macro").to(self.device)

        # Be sure to store experiment details here for collecting results across runs
        self.metrics = {}

    def __call__(self, epochs: int):
        """Trains the model for a given number of epochs."""
        self.global_step = 0  # batches seen
        self.epoch = 0
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        for _ in trange(epochs, desc="Epochs", leave=False, position=self.tqdm_pos):
            self.writer.add_scalar("epoch", self.epoch, self.global_step)
            self._train()
            if self.val_loader is not None:
                self._val()
            self.epoch += 1
            self.checkpoint()
            self.sched.step()
            self.writer.add_scalar("lr", self.sched.get_last_lr()[0], global_step=self.global_step)

        # After training, load the best model (if available), or else the most recent model
        # NOTE - checked by hand that the *.pt file saved in outer loop is same as best_model.pt,
        # and different than checkpoint.pt
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        try:
            ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
        except AttributeError:
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        return self.metrics

    def _run_epoch(self, desc, loader, optim=None):
        self.bal_acc_metric.reset()
        self.acc_metric.reset()
        if optim:
            self.model.train()
        else:
            self.model.eval()

        pbar = tqdm(loader, desc=desc, leave=False, position=self.tqdm_pos + 1)
        total_loss = 0.0
        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            log_probs = self.model(data)
            loss = self.criterion(log_probs, labels)

            if optim:
                optim.zero_grad()
                loss.backward()
                optim.step()

            total_loss += loss.item()
            batch_bal_acc = self.bal_acc_metric(log_probs.argmax(-1), labels)
            batch_acc = self.acc_metric(log_probs.argmax(-1), labels)

            if optim:
                # During training, update metrics each batch
                results = {
                    f"{desc}/batch_loss": float(loss),
                    f"{desc}/batch_acc": batch_acc,
                    f"{desc}/batch_bal_acc": batch_bal_acc,
                }
                pbar.set_postfix({k: f"{v:.3f}" for k, v in results.items()})
                for key, val in results.items():
                    self.writer.add_scalar(key, val, self.global_step)
                self.global_step += 1

        results = {
            f"{desc}/epoch_loss": total_loss / len(loader),
            f"{desc}/epoch_acc": self.acc_metric.compute(),
            f"{desc}/epoch_bal_acc": self.bal_acc_metric.compute(),
        }
        for key, val in results.items():
            self.writer.add_scalar(key, val, self.global_step)
        results["epoch"] = self.epoch
        self.metrics.update(results)

    def _train(self):
        """Training loop"""
        self._run_epoch("train", self.train_loader, self.optim)

    def _val(self):
        self._run_epoch("val", self.val_loader, None)

    def checkpoint(self):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optim_state_dict": self.optim.state_dict(),
        }
        self.ckpt_path = self.results_dir / "checkpoint.pt"
        torch.save(ckpt, self.ckpt_path)
        # TODO - should use "val/epoch_loss" - but there are nans. For now, just use best bal acc.
        # Source of nans is unclear - log_probs from model contain nans in both classes
        # for a single item - but not able to reproduce this by scanning through full train
        # set (including val slice) using model.load("broken.pt") and model.predict(train_x)
        if self.metrics["val/epoch_bal_acc"] > self.best_val_bal_acc:
            self.best_val_bal_acc = self.metrics["val/epoch_bal_acc"]
            self.best_ckpt_path = self.results_dir / "best_model.pt"
            torch.save(ckpt, self.best_ckpt_path)


class Trainer_RNN:
    def __init__(
        self,
        model,
        datamodule: Datamodule,
        lr: float,
        lambda_loss: float,
        model_size: str,
        reward: str,
        results_dir: Path,
        device: torch.device,
        tqdm_pos=0,
    ):
        self.device = device
        self.tqdm_pos = tqdm_pos
        # data
        self.n_classes = datamodule.n_classes
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = None if datamodule.val_set is None else datamodule.val_dataloader()

        # model
        self.model = model.to(device)
        self.optim = AdamW(model.parameters(), lr=lr)
        self.sched = ExponentialLR(self.optim, gamma=0.97)
        self.model_size = model_size

        # Typing Task parameters
        self.n_chars_to_spell = N_CHARS_TO_SPELL
        self.alphabet_len = ALPHABET_LEN
        self.query_selection_method = QUERY_SELECTION_METHOD
        self.trials_per_sequence = TRIALS_PER_SEQUENCE
        self.threshold = DECISION_THRESHOLD
        self.total_attempts = TOTAL_ATTEMPTS

        # reward parameters
        self.gamma = 1.0
        self.lambda_loss = lambda_loss
        self.reward = reward

        # loss functions
        self.criterion = torch.nn.NLLLoss()

        # bookkeeping
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.results_dir)
        self.best_val_acc = -torch.inf
        self.best_val_loss = torch.inf
        self.acc_metric = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        self.bal_acc_metric = Accuracy(task="multiclass", num_classes=self.n_classes, average="macro").to(self.device)

        # Be sure to store experiment details here for collecting results across runs
        self.metrics = {}

    def __call__(self, epochs: int):
        """Trains the model for a given number of epochs."""
        self.global_step = 0  # batches seen
        self.epoch = 0
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        for _ in trange(epochs, desc="Epochs", leave=False, position=self.tqdm_pos):
            self.writer.add_scalar("epoch", self.epoch, self.global_step)
            self.writer.add_scalar("lambda_loss", self.lambda_loss, self.global_step)
            self._train()
            if self.val_loader is not None:
                self._val()
            self.epoch += 1
            self.checkpoint()
            self.sched.step()
            self.writer.add_scalar("lr", self.sched.get_last_lr()[0], global_step=self.global_step)

        # After training, load the best model (if available), or else the most recent model
        # NOTE - checked by hand that the *.pt file saved in outer loop is same as best_model.pt,
        # and different than checkpoint.pt
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        try:
            ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
        except AttributeError:
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        return self.metrics

    def _run_epoch(self, desc, loader, optim=None):
        self.bal_acc_metric.reset()
        self.acc_metric.reset()
        if optim:
            self.model.train()
        else:
            self.model.eval()

        pbar = tqdm(loader, desc=desc, leave=False, position=self.tqdm_pos + 1)
        batch_count = 0
        total_loss, total_loss_action, total_loss_baseline, total_loss_reinforce = 0.0, 0.0, 0.0, 0.0
        predict_all_epoch, target_all_epoch = [], []
        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            target_all, predict_all = [], []
            loss, loss_action_sum, loss_baseline_sum, loss_reinforce_sum = 0.0, 0.0, 0.0, 0.0

            true_letter_idx_all = np.linspace(0, self.alphabet_len - 1, self.alphabet_len, dtype=int)
            true_letter_idx_all = np.random.permutation(true_letter_idx_all)
            target_all.append(true_letter_idx_all)

            for ind in range(self.n_chars_to_spell):
                true_letter_idx = np.array([true_letter_idx_all[ind]])
                log_pi, baselines, R = [], [], []
                if self.model_size == "large":
                    h_t = torch.zeros(32 * self.alphabet_len, dtype=torch.float, device=self.device)
                elif self.model_size == "medium":
                    h_t = torch.zeros(24 * self.alphabet_len, dtype=torch.float, device=self.device)
                elif self.model_size == "small":
                    h_t = torch.zeros(16 * self.alphabet_len, dtype=torch.float, device=self.device)

                # Get prior over alphabet (uniform for now)
                log_probs = np.log(np.ones(self.alphabet_len) / self.alphabet_len)
                attempts_remaining = self.total_attempts
                attempts = 0
                this_letter_labels = []
                this_letter_log_posteriors = []
                this_letter_queried_letter_indices = []
                while attempts_remaining > 0:
                    attempts_remaining -= 1
                    attempts += 1
                    data_attempt, label_attempt, queried_letter_indices_attempt = get_query(
                        query_selection_method=self.query_selection_method,
                        log_probs=log_probs,
                        test_data=data.detach().cpu(),
                        test_labels=labels.detach().cpu(),
                        query_size=self.trials_per_sequence,
                        target_letter_alphabet_idx=true_letter_idx,
                    )
                    this_letter_labels.append(label_attempt.copy())
                    this_letter_queried_letter_indices.append(queried_letter_indices_attempt.copy())

                    data_attempt = torch.from_numpy(data_attempt).float().to(self.device)
                    h_t, b_t, log_probs = self.model(data_attempt, h_t, torch.tensor(queried_letter_indices_attempt))

                    probs = torch.exp(log_probs)
                    this_letter_log_posteriors.append(log_probs.clone())
                    predicted = torch.max(probs, 0)[1]
                    if self.reward == "Linear":
                        r_i = (predicted == torch.from_numpy(true_letter_idx).to(self.device)).float() * (
                            torch.tensor(1 + (attempts_remaining) / self.total_attempts)
                        )
                    elif self.reward == "Rational":
                        r_i = (predicted == torch.from_numpy(true_letter_idx).to(self.device)).float() * (
                            torch.tensor(1 / attempts)
                        )
                    elif self.reward == "InverseSquare":
                        r_i = (predicted == torch.from_numpy(true_letter_idx).to(self.device)).float() * (
                            torch.tensor(1 / (attempts**2))
                        )
                    elif self.reward == "InverseCube":
                        r_i = (predicted == torch.from_numpy(true_letter_idx).to(self.device)).float() * (
                            torch.tensor(1 / (attempts**3))
                        )
                    elif self.reward == "InverseFourth":
                        r_i = (predicted == torch.from_numpy(true_letter_idx).to(self.device)).float() * (
                            torch.tensor(1 / (attempts**4))
                        )
                    R.append((self.gamma**attempts) * (r_i))
                    baselines.append(b_t)
                    log_pi.append(torch.max(log_probs, 0)[0])
                    if attempts_remaining > 0:
                        log_probs = log_probs.detach().cpu().numpy()

                predict_all.append(log_probs.detach().cpu().numpy().argmax(-1))
                R = torch.stack(R).to(self.device)
                baselines = torch.stack(baselines).unsqueeze(0).transpose(1, 0)
                log_pi = torch.stack(log_pi).unsqueeze(0).transpose(1, 0)
                loss_action = self.criterion(log_probs.unsqueeze(0), torch.tensor(true_letter_idx).to(self.device))
                loss_baseline = F.mse_loss(baselines, R)
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=0)

                loss += loss_action + (loss_baseline + loss_reinforce) * self.lambda_loss
                loss_action_sum += loss_action
                loss_baseline_sum += loss_baseline * self.lambda_loss
                loss_reinforce_sum += loss_reinforce * self.lambda_loss

            predict_all_epoch.append(predict_all)
            target_all_epoch.append(target_all)
            loss = loss / (self.n_chars_to_spell)
            loss_action_sum = loss_action_sum / (self.n_chars_to_spell)
            loss_baseline_sum = loss_baseline_sum / (self.n_chars_to_spell)
            loss_reinforce_sum = loss_reinforce_sum / (self.n_chars_to_spell)
            if optim:
                optim.zero_grad()
                loss.backward()
                optim.step()

            total_loss += loss.item()
            total_loss_action += loss_action_sum.item()
            total_loss_baseline += loss_baseline_sum.item()
            total_loss_reinforce += loss_reinforce_sum.item()
            predict_all = torch.tensor(np.array(predict_all)).to(self.device)
            target_all = torch.tensor(np.array(target_all)).flatten().to(self.device)

            batch_acc = self.acc_metric(predict_all, target_all)
            if optim:
                # During training, update metrics each batch
                results = {
                    f"{desc}/batch_loss": float(loss),
                    f"{desc}/batch_action_loss": float(loss_action_sum),
                    f"{desc}/batch_baseline_loss": float(loss_baseline_sum),
                    f"{desc}/batch_reinforce_loss": float(loss_reinforce_sum),
                    f"{desc}/batch_acc": batch_acc,
                }
                pbar.set_postfix({k: f"{v:.3f}" for k, v in results.items()})
                for key, val in results.items():
                    self.writer.add_scalar(key, val, self.global_step)
                self.global_step += 1
            batch_count += 1
        predict_all_epoch = torch.tensor(np.array(predict_all_epoch)).to(self.device)
        target_all_epoch = torch.tensor(np.array(target_all_epoch)).flatten().to(self.device)

        results = {
            f"{desc}/epoch_loss": total_loss / (batch_count),
            f"{desc}/epoch_action_loss": total_loss_action / (batch_count),
            f"{desc}/epoch_baseline_loss": total_loss_baseline / (batch_count),
            f"{desc}/epoch_reinforce_loss": total_loss_reinforce / (batch_count),
            f"{desc}/epoch_acc": self.acc_metric.compute(),
        }
        for key, val in results.items():
            self.writer.add_scalar(key, val, self.global_step)
        results["epoch"] = self.epoch
        self.metrics.update(results)

    def _train(self):
        """Training loop"""
        self._run_epoch("train", self.train_loader, self.optim)

    def _val(self):
        self._run_epoch("val", self.val_loader, None)

    def checkpoint(self):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optim_state_dict": self.optim.state_dict(),
        }
        self.ckpt_path = self.results_dir / "checkpoint_rnn.pt"
        torch.save(ckpt, self.ckpt_path)
        # TODO - should use "val/epoch_loss" - but there are nans. For now, just use best bal acc.
        # Source of nans is unclear - log_probs from model contain nans in both classes
        # for a single item - but not able to reproduce this by scanning through full train
        # set (including val slice) using model.load("broken.pt") and model.predict(train_x)
        if self.metrics["val/epoch_acc"] > self.best_val_acc:
            self.best_val_loss = self.metrics["val/epoch_acc"]
            self.best_ckpt_path = self.results_dir / "best_model.pt"
            torch.save(ckpt, self.best_ckpt_path)
