import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import os
from peft import get_peft_model, LoraConfig, TaskType
import generate_data
from deepspeed.utils.zero_to_fp32 import (
    load_state_dict_from_zero_checkpoint,
    get_fp32_state_dict_from_zero_checkpoint,
)
import torch
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import os, json
from torch.utils.data import Dataset
from gpu_utilities import print_gpu_utilization
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class T5Finetuner(pl.LightningModule):
    def __init__(self, args, train_data, val_data):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(
            # self.args.model_name,
            './hf_model.bin/',
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
        self.validation_step_outputs = []
        # self.get_peft()

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
        source_ids, source_mask, target_ids, target_labels = batch

        # real_input = [
        #     self.tokenizer.decode(
        #         g, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #     )
        #     for g in source_ids
        # ]

        # real_output = [
        #     self.tokenizer.decode(
        #         g, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #     )
        #     for g in target_ids
        # ]

        # generated_ids = self.model.generate(
        #     input_ids=source_ids,
        #     attention_mask=source_mask,
        #     length_penalty=1.0,
        #     early_stopping=True,
        # )

        # guess_output = [
        #     self.tokenizer.decode(
        #         g, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #     )
        #     for g in generated_ids
        # ]
        # print("real_input", real_input)
        # print('real_output', real_output)
        # print('guess_output', guess_output)
        # print()

        loss = self(batch, batch_idx)[0]
        self.validation_step_outputs.append(loss)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        print(epoch_average)
        with open('output.txt', 'a') as file:
            file.write(str(epoch_average))
            file.write('\n')
        self.validation_step_outputs.clear()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=50,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.args.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=50,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.trainer.model.parameters(), lr=self.args.lr, weight_decay=0.01
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
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
            max_length=16,
            truncation=True,
            # pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [text],
            max_length=16,
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


def start_training(args):
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
        strategy="deepspeed_stage_3",  # https://lightning.ai/docs/pytorch/latest/extensions/strategy.html#:~:text=The%20Strategy%20in%20PyTorch%20Lightning,%2C%20broadcast%2C%20and%20so%20on.
        check_val_every_n_epoch=1,
        logger=TensorBoardLogger(
            os.path.join(args.output, "logs"), name=args.model_name
        ),
        log_every_n_steps=1,
    )
    trainer.fit(summarizer)

    ckpt_path = f"./output/logs/{args.model_name}/version_{trainer.logger.version}/checkpoints/epoch={trainer.current_epoch-1}-step={trainer.global_step}.ckpt"
    convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, "lightning_model.pt")
    
    # state_dict = torch.load("lightning_model.pt")["state_dict"]
    # torch.save(state_dict, "state_dict.pt")
    training_model = T5Finetuner.load_from_checkpoint("lightning_model.pt")
    # training_model.load_state_dict(state_dict)
    
    training_model.model.save_pretrained("hf_model.bin")

    # training_model.load_from_checkpoint(ckpt_path)
    # training_model.model.save_pretrained("single_gpu.bin")
    # tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir="./cache/")
    
    # train_data = DataLoader(
    #     train_data,
    #     batch_size=4,
    #     num_workers=1,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    # )
    # for batch in train_data:
    #     source_ids, source_mask, target_ids, target_labels = batch

    #     generated_ids = model.generate(
    #         input_ids=source_ids,
    #         attention_mask=source_mask,
    #         length_penalty=1.0,
    #         early_stopping=True,
    #     )

    #     real_input = [
    #         tokenizer.decode(
    #             g, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #         )
    #         for g in source_ids
    #     ]

    #     guess_output = [
    #         tokenizer.decode(
    #             g, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #         )
    #         for g in generated_ids
    #     ]

    #     real_output = [
    #         tokenizer.decode(
    #             g, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #         )
    #         for g in target_ids
    #     ]
    #     print("Real input:", real_input)
    #     print("Guess Output:", guess_output)
    #     print("Real Output:", real_output)
    #     print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="./data/")
    parser.add_argument("-c", "--cache", default="./cache/")
    parser.add_argument("-o", "--output", default="./output/")
    parser.add_argument("-t", "--train_size", default=20_000)
    parser.add_argument("-v", "--val_size", default=2_000)
    parser.add_argument("-m", "--model_name", default="t5-base")
    parser.add_argument("-l", "--lr", default=1e-07)
    parser.add_argument("-e", "--epochs", default=10)
    parser.add_argument("-b", "--batch_size", default=16)
    args = parser.parse_args()
    start_training(args)
