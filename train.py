import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
import os
from peft import get_peft_model, LoraConfig, TaskType
import generate_data


import torch

# from fairscale.nn import (
#     checkpoint_wrapper,
# )  # https://neptune.ai/blog/multi-gpu-model-training-monitoring-and-optimizing
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import os, json
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class T5Finetuner(pl.LightningModule):
    def __init__(self, args, train_data, val_data):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.args.model_name,
            cache_dir=args.cache,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, cache_dir=args.cache, use_fast=True
        )
        self.model.gradient_checkpointing_enable()
        self.cache_dir = args.cache

        self.train_data, self.val_data = train_data, val_data

        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.get_peft()

    def get_peft(self):
        self.model.enable_input_require_grads()
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def forward(self, batch, batch_idx):
        source_ids, source_mask, target_ids, target_labels = batch

        return self.model(
            input_ids=source_ids,
            attention_mask=source_mask,
            decoder_input_ids=target_ids,
            labels=target_labels,
        )

    def training_step(self, batch, batch_idx):
        # accumulation: https://lightning.ai/docs/fabric/latest/advanced/gradient_accumulation.html
        loss = self(batch, batch_idx)[0]
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]

        return {
            "loss": loss,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.args.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.trainer.model.parameters(), lr=self.args.lr, weight_decay=0.01
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=self.args.epochs
            * len(self.train_data)
            / self.args.batch_size,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class StreamingDataset(Dataset):
    def __init__(self, path, model_name):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=args.cache, use_fast=True
        )

    def __len__(self):
        return len(os.listdir(self.path))

    def encode_text(self, context, text):
        ctext = str(context)
        ctext = " ".join(ctext.split())
        text = str(text)
        text = " ".join(text.split())
        source = self.tokenizer.batch_encode_plus(
            [ctext],
            max_length=512,
            truncation=True,
            # pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [text],
            max_length=150,
            truncation=True,
            # pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )
        y = target["input_ids"]
        target_id = y[:, :-1].contiguous()
        target_label = y[:, 1:].clone().detach()
        target_label[
            y[:, 1:] == self.tokenizer.pad_token_id
        ] = -100  # in case the labels are not provided, empty string
        # return torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])
        return source["input_ids"], source["attention_mask"], target_id, target_label

    def __getitem__(self, idx):
        file_path = os.path.join(self.path, str(idx) + ".json")
        with open(file_path, "r") as infile:
            data = json.load(infile)
        number, words = str(data["number"]), data["words"]
        return self.encode_text(number, words)


def collate_fn(batch):
    input_ids = torch.stack([torch.flatten(x[0]) for x in batch])
    sequence_mask = torch.stack([torch.flatten(x[1]) for x in batch])
    target_ids = torch.stack([torch.flatten(x[2]) for x in batch])
    target_label = torch.stack([torch.flatten(x[3]) for x in batch])
    return input_ids, sequence_mask, target_ids, target_label


def train(args):
    generate_data.main(args.train_size, args.val_size)
    train_data = StreamingDataset(os.path.join(args.data, "train"), args.model_name)
    val_data = StreamingDataset(os.path.join(args.data, "val"), args.model_name)

    summarizer = T5Finetuner(args, train_data, val_data)
    # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html?highlight=gradient%20accumulation
    # checkpointing: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#activation-checkpointing
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        accumulate_grad_batches=4,
        strategy="deepspeed_stage_2",  # https://lightning.ai/docs/pytorch/latest/extensions/strategy.html#:~:text=The%20Strategy%20in%20PyTorch%20Lightning,%2C%20broadcast%2C%20and%20so%20on.
        check_val_every_n_epoch=1,
        logger=TensorBoardLogger(
            os.path.join(args.output, "logs"), name=args.model_name
        ),
    )
    trainer.fit(summarizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="./data/")
    parser.add_argument("-c", "--cache", default="./cache/")
    parser.add_argument("-o", "--output", default="./output/")
    parser.add_argument("-t", "--train_size", default=10_000)
    parser.add_argument("-v", "--val_size", default=2000)
    parser.add_argument("-m", "--model_name", default="t5-small")
    parser.add_argument("-l", "--lr", default=1e-5)
    parser.add_argument("-e", "--epochs", default=15)
    parser.add_argument("-b", "--batch_size", default=16)
    args = parser.parse_args()
    train(args)
