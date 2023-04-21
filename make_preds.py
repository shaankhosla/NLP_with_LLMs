import torch
from train import T5Finetuner
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration


class TextGenerate:
    def __init__(self, model_name, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache/")

        self.model = T5ForConditionalGeneration.from_pretrained(
                    't5-small',cache_dir='./cache/',

                )
        self.model.load_state_dict(torch.load('./model.bin'), strict=False)

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
        return source["input_ids"], source["attention_mask"], target_id, target_label


    def generate_prediction(
        self, ctext, summ_len=150, beam_search=2, repetition_penalty=2.5
    ):
        profile_ids, _, _, _ = self.encode_text(ctext, "")
        self.model.eval()
        with torch.no_grad():
            pred_ids = self.model.generate(profile_ids)
            
        predictions = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in pred_ids
        ]
        return predictions


if __name__ == "__main__":
    model = TextGenerate(
        "t5-small",
        "./model.bin",
    )
    output = model.generate_prediction(
        """4463
        """
    )
    print(output)
