"""
Script for self-supervised pretraining/supervised finetuning BarcodeMamba, including train and evaluate.
"""

import os
from einops import rearrange
import hydra
import hydra.core.hydra_config
import lightning as pl
import torch
from torchmetrics import MetricCollection
from utils.barcode_mamba import BarcodeMamba
import torch.optim as optim
from utils.ssm_dataset import get_dataloader
from utils.train_utils import (
    NumTokens,
    Perplexity,
    TimmCosineLRScheduler,
    accuracy,
    cross_entropy,
)
from omegaconf import OmegaConf as o
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import logging

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
try:
    o.register_new_resolver("eval", eval)
    o.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
except Exception:
    pass


class BarcodeMamba_lightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BarcodeMamba(**config.model, use_head=self.config.dataset.phase)
        self.save_hyperparameters(config)
        self.loss = cross_entropy
        self.accuracy = accuracy
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        if self.config.dataset.phase == "pretrain":
            metric_collection = MetricCollection(
                {"perplexity": Perplexity(), "num_tokens": NumTokens()}
            )
            self.train_metrics = metric_collection.clone(prefix="train/")
            self.val_metrics = metric_collection.clone(prefix="val/")
            self.test_metrics = metric_collection.clone(prefix="test/")
        elif self.config.dataset.phase == "finetune":
            self.validation_step_acc = []
            self.test_step_acc = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.config.optimizer)
        scheduler = {
            "scheduler": TimmCosineLRScheduler(
                **self.config.scheduler, optimizer=optimizer
            ),
            "interval": self.config.train.interval,
            "monitor": self.config.train.monitor,
            "name": "trainer/lr",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self(x)
        x = rearrange(x, "... C -> (...) C")
        y = rearrange(y, "... -> (...)")
        loss = self.loss(x, y)
        self.log("train/loss_step", loss, on_epoch=False, on_step=True, sync_dist=True)
        self.training_step_outputs.append(loss)
        if self.config.dataset.phase == "pretrain":
            self.train_metrics(x, y, loss=loss)
            self.log_dict(
                self.train_metrics,
                on_step=True,
                on_epoch=True,
                # prog_bar=True,
                sync_dist=True,
            )
        return loss

    def on_train_epoch_end(self):
        loss_epoch = torch.stack(self.training_step_outputs).mean()
        self.log("train/loss_epoch", loss_epoch, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self(x)
        x = rearrange(x, "... C -> (...) C")
        y = rearrange(y, "... -> (...)")
        loss = self.loss(x, y)
        self.log(
            "val/loss_step",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )
        self.validation_step_outputs.append(loss)
        if self.config.dataset.phase == "pretrain":
            self.val_metrics(x, y, loss=loss)
            self.log_dict(
                self.val_metrics,
                on_step=True,
                on_epoch=True,
                # prog_bar=True,
                sync_dist=True,
            )
        elif self.config.dataset.phase == "finetune":
            acc = self.accuracy(x, y)
            self.log(
                "val/accuracy_step",
                acc,
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )
            self.validation_step_acc.append(acc)

        return loss

    def on_validation_epoch_end(self):
        loss_epoch = torch.stack(self.validation_step_outputs).mean()
        self.log(
            "val/loss_epoch",
            loss_epoch,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()

        if self.config.dataset.phase == "finetune":
            acc_epoch = torch.stack(self.validation_step_acc).mean()
            self.log(
                "val/acc_epoch",
                acc_epoch,
                sync_dist=True,
            )
            self.validation_step_acc.clear()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self(x)
        x = rearrange(x, "... C -> (...) C")
        y = rearrange(y, "... -> (...)")
        loss = self.loss(x, y)
        self.log(
            "test/loss_step",
            loss,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )
        self.test_step_outputs.append(loss)
        if self.config.dataset.phase == "pretrain":
            self.test_metrics(x, y, loss=loss)
            self.log_dict(
                self.test_metrics,
                on_step=True,
                on_epoch=True,
                # prog_bar=True,
                sync_dist=True,
            )
        elif self.config.dataset.phase == "finetune":
            acc = self.accuracy(x, y)
            self.log(
                "test/accuracy_step",
                acc,
                on_epoch=False,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.test_step_acc.append(acc)

        return loss

    def on_test_epoch_end(self):
        loss_epoch = torch.stack(self.test_step_outputs).mean()
        self.log(
            "test/loss_epoch",
            loss_epoch,
            sync_dist=True,
        )
        self.test_step_outputs.clear()

        if self.config.dataset.phase == "finetune":
            acc_epoch = torch.stack(self.test_step_acc).mean()
            self.log(
                "test/acc_epoch",
                acc_epoch,
                sync_dist=True,
            )
            self.test_step_acc.clear()

    def train_dataloader(self):
        return get_dataloader(config=self.config, phase="train")

    def val_dataloader(self):
        return get_dataloader(config=self.config, phase="val")

    def test_dataloader(self):
        return get_dataloader(config=self.config, phase="test")


def train(cfg: o):
    if cfg.train.seed is not None:
        pl.seed_everything(cfg.train.seed, workers=True)
    if cfg.train.logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project="barcode-mamba", name=cfg.train.run_name)
    else:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger("./")
    trainer = pl.Trainer(
        **cfg.trainer,
        # enable_progress_bar=False,
        logger=logger,
        callbacks=[
            ModelCheckpoint(**cfg.model_checkpoint),
            EarlyStopping(
                monitor=cfg.train.monitor,
                patience=3,
                verbose=True,
                mode=cfg.train.mode,
            ),
        ],
    )
    lightning_model = BarcodeMamba_lightning(cfg)
    if cfg.train.get("pretrained_model_path", None) is not None:
        lightning_model.load_state_dict(
            torch.load(cfg.train.pretrained_model_path)["state_dict"],
            strict=cfg.train.pretrained_model_strict_load,
        )
        logging.info(f"loaded pretrained_model_path: {cfg.train.pretrained_model_path}")
    if cfg.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(lightning_model)

    if cfg.train.ckpt is not None:
        trainer.fit(lightning_model, ckpt_path=cfg.train.ckpt)
    else:
        trainer.fit(lightning_model)
    if cfg.train.test:
        trainer.test(lightning_model)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: o):
    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.chdir(hydra_output_path)
    train(config)


if __name__ == "__main__":
    main()
