import tqdm
import math
import wandb
import torch
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Callable, List
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc


def write_dictconfig(d, f, child: bool = False, ntab=0):
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def initialize_wandb(
    cfg: DictConfig,
    tags: Optional[List] = None,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if tags == None:
        tags = []
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.username,
        name=cfg.wandb.exp_name,
        group=cfg.wandb.group,
        dir=cfg.wandb.dir,
        config=config,
        tags=tags,
    )
    config_file_path = Path(run.dir, "run_config.yaml")
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_binary_metrics(probs: np.array(float), preds: List[int], labels: List[int]):
    labels = np.asarray(labels)
    acc = metrics.accuracy_score(labels, preds)
    auc = metrics.roc_auc_score(labels, probs)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds)
    metrics_dict = {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
    }
    return metrics_dict


def get_metrics(
    probs: np.array(float),
    preds: List[int],
    labels: List[int],
    multi_class: str = "ovr",
):
    labels = np.asarray(labels)
    auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
    quadratic_weighted_kappa = metrics.cohen_kappa_score(
        labels, preds, weights="quadratic"
    )
    metrics_dict = {"auc": auc, "kappa": quadratic_weighted_kappa}
    return metrics_dict


def collate_features(batch, label_type: str = "int"):
    idx = torch.LongTensor([item[0] for item in batch])
    # feature = torch.vstack([item[1] for item in batch])
    feature = torch.cat([item[1] for item in batch], dim=0)
    if label_type == "float":
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[2] for item in batch])
    return [idx, feature, label]


def get_roc_auc_curve(probs: np.array(float), labels: List[int], log_to_wandb: bool = False):
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    auc = metrics.roc_auc_score(labels, probs)
    fig = plt.figure(dpi=600)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend(loc="lower right")
    if log_to_wandb:
        img = wandb.Image(fig)
    plt.close()
    return fig


def update_log_dict(
    prefix, results, log_dict, step: Optional[str] = "step", to_log: Optional[List["str"]] = None
):
    if not to_log:
        to_log = list(results.keys())
    for r, v in results.items():
        if r in to_log:
            wandb.define_metric(f"{prefix}/{r}", step_metric=step)
            log_dict.update({f"{prefix}/{r}": v})


def resume_from_checkpoint(ckpt_path, **kwargs):
    """
    Re-start from checkpoint
    """
    if not Path(ckpt_path).is_file():
        return
    print(f"Found checkpoint at {ckpt_path}")

    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    f"=> loaded '{key}' from checkpoint: '{ckpt_path}' with msg {msg}"
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print(f"=> loaded '{key}' from checkpoint: '{ckpt_path}'")
                except ValueError:
                    print(f"=> failed to load '{key}' from checkpoint: '{ckpt_path}'")
        else:
            print(f"=> key '{key}' not found in checkpoint: '{ckpt_path}'")
    return epoch


def make_weights_for_balanced_classes(dataset):
    n_samples = len(dataset)
    weight_per_class = []
    for c in range(dataset.num_classes):
        w = n_samples * 1.0 / len(dataset.class_2_id[c])
        weight_per_class.append(w)
    weight = []
    for idx in range(len(dataset)):
        y = dataset.get_label(idx)
        weight.append(weight_per_class[y])
    return torch.DoubleTensor(weight)


class OptimizerFactory:
    def __init__(
        self,
        name: str,
        params: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):

        if name == "adam":
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            self.optimizer = optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise KeyError(f"{name} not supported")

    def get_optimizer(self):
        return self.optimizer


class SchedulerFactory:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: Optional[dict] = None,
    ):

        self.scheduler = None
        self.name = params.name
        if self.name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=params.step_size, gamma=params.gamma
            )
        elif self.name == "cosine":
            assert (
                params.T_max != -1
            ), "T_max parameter must be specified! If you dont know what to use, plug in nepochs"
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                params.T_max, eta_min=params.eta_min
            )
        elif self.name == "reduce_lr_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params.mode,
                factor=params.factor,
                patience=params.patience,
                min_lr=params.min_lr,
            )
        elif self.name:
            raise KeyError(f"{self.name} not supported")

    def get_scheduler(self):
        return self.scheduler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        tracking: str,
        min_max: str,
        patience: int = 20,
        min_epoch: int = 50,
        checkpoint_dir: Optional[Path] = None,
        save_all: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.tracking = tracking
        self.min_max = min_max
        self.patience = patience
        self.min_epoch = min_epoch
        self.checkpoint_dir = checkpoint_dir
        self.save_all = save_all
        self.verbose = verbose

        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, save_dict, results):

        score = results[self.tracking]
        if self.min_max == "min":
            score = -1 * score

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            fname = f"best.pt"
            torch.save(save_dict, Path(self.checkpoint_dir, fname))
            self.counter = 0

        elif score < self.best_score:
            self.counter += 1
            if epoch <= self.min_epoch + 1 and self.verbose:
                print(
                    f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}"
                )
            elif self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience and epoch > self.min_epoch:
                self.early_stop = True

        if self.save_all:
            fname = f"epoch_{epoch}.pt"
            torch.save(save_dict, Path(self.checkpoint_dir, fname))

        # override latest
        torch.save(save_dict, Path(self.checkpoint_dir, "latest.pt"))


def collate_survival_tensors(batch, label_type: str = "int", agg_method: str = "max_slide"):
    idx = torch.LongTensor([item[0] for item in batch])
    if agg_method == "max_slide":
        st = [item[1] for item in batch]
    elif agg_method == "self_att":
        st = [item[1] for item in batch]
    if label_type == "float":
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[2] for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    censorship = torch.FloatTensor([item[4] for item in batch])
    return [idx, st, label, event_time, censorship]


def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.RandomSampler(dataset)
    if weighted_sampling:
        weights = make_weights_for_balanced_classes(dataset)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            len(weights),
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Train - Epoch {epoch}"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        for i, batch in enumerate(t):

            optimizer.zero_grad()
            idx, x, label = batch
            x, label = x.to(device, non_blocking=True), label.to(
                device, non_blocking=True
            )
            logits = model(x)
            loss = criterion(logits, label)

            loss_value = loss.item()
            epoch_loss += loss_value

            loss.backward()
            optimizer.step()

            pred = torch.topk(logits, 1, dim=1)[1]
            preds.extend(pred[:, 0].clone().tolist())

            prob = F.softmax(logits, dim=1).cpu().detach().numpy()
            probs = np.append(probs, prob, axis=0)

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:, 1], preds, labels)
        roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(probs, preds, labels)

    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Tune - Epoch {epoch}"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)
                loss = criterion(logits, label)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:, 1], preds, labels)
        roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(probs, preds, labels)

    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Test"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:, 1], preds, labels)
        roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(probs, preds, labels)

    results.update(metrics)

    return results


def train_survival(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    batch_size: Optional[int] = 1,
    agg_method: str = "max_slide",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.RandomSampler(dataset)
    collate_fn = partial(collate_survival_tensors, label_type="int")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Train - Epoch {epoch}"),
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        for i, batch in enumerate(t):

            idx, x, label, event_time, c = batch
            label, c = label.to(device, non_blocking=True), c.to(device, non_blocking=True)
            if agg_method == "max_slide":
                x = x.to(device, non_blocking=True)
            elif agg_method == "self_att":
                x = [t.to(device, non_blocking=True) for t in x]

            logits = model(x)                           # [1, nbins]
            hazards = torch.sigmoid(logits)             # [1, nbins]
            surv = torch.cumprod(1 - hazards, dim=1)    # [1, nbins]

            loss = criterion(hazards, surv, label, c)

            loss_value = loss.item()
            epoch_loss += loss_value

            risk = -torch.sum(surv, dim=1).detach()     # [1]
            risk_scores.append(risk.item())
            censorships.append(c.item())
            event_times.append(event_time.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    dataset.patient_df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1-c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_survival(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    batch_size: Optional[int] = 1,
    agg_method: str = "max_slide",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)
    collate_fn = partial(collate_survival_tensors, label_type="int")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Tune - Epoch {epoch}"),
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, label, event_time, c = batch
                label, c = label.to(device, non_blocking=True), c.to(device, non_blocking=True)
                if agg_method == "max_slide":
                    x = x.to(device, non_blocking=True)
                elif agg_method == "self_att":
                    x = [t.to(device, non_blocking=True) for t in x]

                logits = model(x)
                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                loss = criterion(hazards, surv, label, c, alpha=0)
                epoch_loss += loss.item()

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censorships.append(c.item())
                event_times.append(event_time.item())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    dataset.patient_df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1-c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index
    results["risks"] = risk_scores

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test_survival(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: Optional[int] = 1,
    agg_method: str = "max_slide",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)
    collate_fn = partial(collate_survival_tensors, label_type="int")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Test"),
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, label, event_time, c = batch
                label, c = label.to(device, non_blocking=True), c.to(device, non_blocking=True)
                if agg_method == "max_slide":
                    x = x.to(device, non_blocking=True)
                elif agg_method == "self_att":
                    x = [t.to(device, non_blocking=True) for t in x]

                logits = model(x)
                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censorships.append(c.item())
                event_times.append(event_time.item())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    dataset.patient_df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1-c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index

    return results
