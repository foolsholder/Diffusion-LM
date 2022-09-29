import torch
from transformers import BloomTokenizerFast, BloomForCausalLM

def dict_to_device(dct, device):
    return {k:v.to(device) for k, v in dct.items()}

class BloomMetric:
    def __init__(self, device="cpu"):
        self.name = "bigscience/bloom-7b1"
        self.model = BloomForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, batch_texts):
        inputs = self.tokenizer("\n\n".join(batch_texts), return_tensors="pt", max_length=128)
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        return loss.item()


from transformers import GPT2Tokenizer, GPT2LMHeadModel


class GPTMetric:
    def __init__(self, device="cpu"):
        self.name = "gpt2"
        self.model = GPT2LMHeadModel.from_pretrained(self.name).eval().to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, batch_texts):
        inputs = self.tokenizer("\n\n".join(batch_texts), return_tensors="pt", max_length=128)
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        return loss.item()

from transformers import T5TokenizerFast, T5ForConditionalGeneration


class T5Metric:
    def __init__(self, device):
        self.t5_name = "t5-3b"
        self.model = T5ForConditionalGeneration.from_pretrained(self.t5_name).eval().to(device)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.t5_name)
        self.device = device

    @torch.no_grad()
    def __call__(self, batch_texts):
        condition = [""] * len(batch_texts)
        encoding = self.tokenizer(condition, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        labels = self.tokenizer(
            batch_texts,
            padding="longest",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        #labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100

        loss = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels).loss.detach().cpu()
        return loss.item()
