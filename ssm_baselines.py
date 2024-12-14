import os
import hydra
from einops import rearrange
import lightning as pl
import torch
import torch.optim as optim
from torch import nn
from utils.seq_decoder import SequenceDecoder
from lightning.pytorch.loggers import WandbLogger
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from torch.utils.data import DataLoader
from utils.ssm_dataset import TokenizedDNADatasetForBaselines
from utils.train_utils import TimmCosineLRScheduler, accuracy, cross_entropy


def get_dataloader(config, tokenizer, phase="train"):
    assert phase in ["train", "val", "test"]
    if phase == "train":
        data_path = os.path.join(config.dataset.input_path, "supervised_train.csv")
    elif phase == "val":
        data_path = os.path.join(config.dataset.input_path, "supervised_val.csv")
    elif phase == "test":
        data_path = os.path.join(config.dataset.input_path, "supervised_test.csv")
    dataset = TokenizedDNADatasetForBaselines(data_path, config.dataset, tokenizer)
    return DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        pin_memory=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        shuffle=False,
    )


def get_ssm_hf_model_tokenizer(model_name, checkpoint):
    assert model_name in ["hyenadna", "mambadna"]
    if model_name == "hyenadna":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "mambadna":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True)
        model = nn.Sequential(*list(model.children())[:-1])
    return model, tokenizer


class FinetuneModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrained_model, self.tokenizer = get_ssm_hf_model_tokenizer(
            config.model_name, config.checkpoint
        )
        if config.classification_dim is None:
            if config.model_name == "hyenadna":
                self.head = SequenceDecoder(
                    d_model=256, d_output=config.n_classes, l_output=0, mode="pool"
                )
            elif config.model_name == "mambadna":
                if "ph" in config.checkpoint:
                    self.head = SequenceDecoder(
                        d_model=256, d_output=config.n_classes, l_output=0, mode="pool"
                    )
                else:
                    self.head = SequenceDecoder(
                        d_model=512, d_output=config.n_classes, l_output=0, mode="pool"
                    )
        else:
            self.head = SequenceDecoder(
                d_model=config.classification_dim,
                d_output=config.n_classes,
                l_output=0,
                mode="pool",
            )
        self.save_hyperparameters(config)
        self.loss = cross_entropy
        self.accuracy = accuracy
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_acc = []
        self.test_step_acc = []

    def forward(self, *args, **kwargs):
        out = self.pretrained_model(*args, **kwargs)
        if isinstance(out, BaseModelOutputWithNoAttention):
            out = out["last_hidden_state"]
        out = self.head(out)
        return out

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.config.optimizer)
        scheduler = {
            "scheduler": TimmCosineLRScheduler(
                **self.config.scheduler, optimizer=optimizer
            ),
            "interval": "step",
            "monitor": "val/loss_epoch",
            "name": "trainer/lr",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx=0):
        x, y = batch
        x = self(x)
        x = rearrange(x, "... C -> (...) C")
        y = rearrange(y, "... -> (...)")
        loss = self.loss(x, y)
        self.log("train/loss_step", loss, on_epoch=False, on_step=True, sync_dist=True)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss_epoch = torch.stack(self.training_step_outputs).mean()
        self.log("train/loss_epoch", loss_epoch, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
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

        acc_epoch = torch.stack(self.validation_step_acc).mean()
        self.log(
            "val/acc_epoch",
            acc_epoch,
            sync_dist=True,
        )
        self.validation_step_acc.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
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
        acc_epoch = torch.stack(self.test_step_acc).mean()
        self.log(
            "test/acc_epoch",
            acc_epoch,
            sync_dist=True,
        )
        self.test_step_acc.clear()

    def train_dataloader(self):
        return get_dataloader(
            config=self.config, tokenizer=self.tokenizer, phase="train"
        )

    def val_dataloader(self):
        return get_dataloader(config=self.config, tokenizer=self.tokenizer, phase="val")

    def test_dataloader(self):
        return get_dataloader(
            config=self.config, tokenizer=self.tokenizer, phase="test"
        )


@hydra.main(version_base=None, config_path="configs", config_name="baseline_config")
def finetune_baseline(config):
    logger_run_name = f"{str(config.checkpoint).replace('/','--')}-finetune"
    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.chdir(hydra_output_path)
    lightning_model = FinetuneModel(config=config)
    logger = WandbLogger(project="barcode-mamba-baselines", name=logger_run_name)
    trainer = pl.Trainer(
        **config.trainer,
        # enable_progress_bar=False,
        logger=logger,
    )
    if not config.test:
        trainer.fit(lightning_model)
        trainer.test(lightning_model)
        trainer.save_checkpoint("last.ckpt")
    else:
        assert config.ckpt is not None
        lightning_model.load_state_dict(
            torch.load(config.ckpt)["state_dict"], strict=False
        )
        trainer.test(lightning_model)


if __name__ == "__main__":
    finetune_baseline()
