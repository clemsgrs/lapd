import os
import time
import tqdm
import timm
import wandb
import torch
import hydra
import datetime
import pandas as pd

from pathlib import Path
from omegaconf import DictConfig

from source.components import LossFactory
from source.dataset import MaxContourTensorSurvivalDataset, ppcess_survival_data
from source.utils import (
    initialize_wandb,
    train_survival,
    tune_survival,
    test_survival,
    compute_time,
    update_log_dict,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(
    version_base="1.2.0", config_path="config", config_name="survival"
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results")
    result_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_classes = cfg.nbins
    criterion = LossFactory(cfg.task, cfg.loss).get_loss()

    model = timm.create_model(cfg.model.arch, pretrained=False, num_classes=num_classes, in_chans=cfg.tile_emb_size)
    # print(model)

    print("Loading data")
    dfs = {}
    tiles_df = pd.read_csv(cfg.tiles_csv)
    for p in ["train", "tune", "test"]:
        df_path = Path(cfg.fold_dir, f"{p}.csv")
        df = pd.read_csv(df_path)
        df = pd.merge(df, tiles_df, on='slide_id')
        df['partition'] = [p] * len(df)
        dfs[p] = df

    if cfg.training.pct:
        print(f"Training on {cfg.training.pct*100}% of the data")
        dfs["train"] = dfs["train"].sample(frac=cfg.training.pct).reset_index(drop=True)

    df = pd.concat([df for df in dfs.values()], ignore_index=True)
    patient_df, slide_df = ppcess_survival_data(df, cfg.label_name, nbins=cfg.nbins)

    patient_dfs, slide_dfs = {}, {}
    for p in ["train", "tune", "test"]:
        patient_dfs[p] = patient_df[patient_df.partition == p].reset_index(drop=True)
        slide_dfs[p] = slide_df[slide_df.partition == p]

    print(f"Initializing training dataset")
    train_dataset = MaxContourTensorSurvivalDataset(
        patient_dfs["train"],
        slide_dfs["train"],
        features_dir,
        cfg.tile_size,
        cfg.tile_fmt,
        cfg.tile_emb_size,
        cfg.label_name,
    )
    print(f"Initializing tuning dataset")
    tune_dataset = MaxContourTensorSurvivalDataset(
        patient_dfs["tune"],
        slide_dfs["tune"],
        features_dir,
        cfg.tile_size,
        cfg.tile_fmt,
        cfg.tile_emb_size,
        cfg.label_name,
    )
    print(f"Initializing testing dataset")
    test_dataset = MaxContourTensorSurvivalDataset(
        patient_dfs["test"],
        slide_dfs["test"],
        features_dir,
        cfg.tile_size,
        cfg.tile_fmt,
        cfg.tile_emb_size,
        cfg.label_name,
    )

    print("Configuring optimmizer & scheduler")
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OptimizerFactory(
        cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
    ).get_optimizer()
    scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

    early_stopping = EarlyStopping(
        cfg.early_stopping.tracking,
        cfg.early_stopping.min_max,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=checkpoint_dir,
        save_all=cfg.early_stopping.save_all,
    )

    stop = False
    start_time = time.time()

    with tqdm.tqdm(
        range(cfg.nepochs),
        desc=(f"LAPD Training"),
        unit=" slide",
        ncols=100,
        leave=True,
    ) as t:

        for epoch in t:

            epoch_start_time = time.time()
            if cfg.wandb.enable:
                log_dict = {"epoch": epoch+1}

            train_results = train_survival(
                epoch+1,
                model,
                train_dataset,
                optimizer,
                criterion,
                batch_size=cfg.training.batch_size,
            )

            if cfg.wandb.enable:
                update_log_dict("train", train_results, log_dict, to_log=cfg.wandb.to_log)
            train_dataset.patient_df.to_csv(Path(result_dir, f"train_{epoch}.csv"), index=False)

            if epoch % cfg.tuning.tune_every == 0:

                tune_results = tune_survival(
                    epoch+1,
                    model,
                    tune_dataset,
                    criterion,
                    batch_size=cfg.tuning.batch_size,
                )

                if cfg.wandb.enable:
                    update_log_dict("tune", tune_results, log_dict, to_log=cfg.wandb.to_log)
                tune_dataset.patient_df.to_csv(Path(result_dir, f"tune_{epoch}.csv"), index=False)

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            lr = cfg.optim.lr
            if scheduler:
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
            if cfg.wandb.enable:
                log_dict.update({"train/lr": lr})

            # logging
            if cfg.wandb.enable:
                wandb.log(log_dict, step=epoch+1)

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            tqdm.tqdm.write(
                f"End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
            )

            if stop:
                tqdm.tqdm.write(
                    f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                )
                break

    if cfg.testing.run_testing:
        # load best model
        best_model_fp = Path(checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}_model.pt")
        if cfg.wandb.enable:
            wandb.save(str(best_model_fp))
        best_model_sd = torch.load(best_model_fp)
        model.load_state_dict(best_model_sd)

        test_results = test_survival(
            model,
            test_dataset,
            batch_size=1,
        )
        test_dataset.patient_df.to_csv(Path(result_dir, f"test.csv"), index=False)

        for r, v in test_results.items():
            if r == "c-index":
                v = round(v, 3)
            if r in cfg.wandb.to_log and cfg.wandb.enable:
                wandb.log({f"test/{r}": v})

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    main()
