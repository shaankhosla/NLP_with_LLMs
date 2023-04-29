import torch
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration


class TextGenerate:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir="./cache/")

        self.model = T5ForConditionalGeneration.from_pretrained('./hf_model.bin')
        # self.model = T5ForConditionalGeneration.from_pretrained("./single_gpu.bin")

        # self.model = T5ForConditionalGeneration.from_pretrained(
        #     "t5-small", cache_dir="./cache/"
        # )
        # state_dict = torch.load("state_dict.pt")
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if "model." in k:
        #         new_k = k.replace("model.", "")
        #         new_state_dict[new_k] = v
        #     else:
        #         new_state_dict[k] = v
        # self.model.load_state_dict(new_state_dict, strict=True)

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

    def generate_prediction(
        self, ctext, summ_len=16, beam_search=10, repetition_penalty=2.5
    ):
        input_ids = self.tokenizer(ctext, return_tensors="pt").input_ids
 
        generated_ids = self.model.generate(input_ids, do_sample=True, 
            max_length=summ_len, 
            top_k=beam_search, 
            temperature=0.7
        )

        summary = self.tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
        return summary



if __name__ == "__main__":
    custom_model = TextGenerate()
    output = custom_model.generate_prediction("5")
    print(output)
    output = custom_model.generate_prediction("-8")
    print(output)
    output = custom_model.generate_prediction("11")
    print(output)
    output = custom_model.generate_prediction("24")
    print(output)
    output = custom_model.generate_prediction("-112")
    print(output)
    output = custom_model.generate_prediction("-236")
    print(output)
    output = custom_model.generate_prediction("-7965")
    print(output)
    output = custom_model.generate_prediction("32043")
    print(output)
    output = custom_model.generate_prediction("34986")
    print(output)
    output = custom_model.generate_prediction("430895")
    print(output)
    output = custom_model.generate_prediction("435641")
    print(output)
    output = custom_model.generate_prediction("-43968")
    print(output)
    output = custom_model.generate_prediction("-328493")
    print(output)
    output = custom_model.generate_prediction("-352501")
    print(output)
    
    
