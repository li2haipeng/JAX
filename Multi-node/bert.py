# -*- coding: utf-8 -*-
from transformers import AutoModelForMaskedLM
from transformers import pipeline
import math
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoConfig, BertConfig, BertTokenizer
from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from pathlib import Path
import time
import torch
import sys
import os
os.environ['NCCL_DEBUG'] = 'TRACE'

args = sys.argv

language = "is"
model_config = "roberta-base"

model_dir = model_config + f"-pretrained-{language}"

# config = BertConfig()

Path(model_dir).mkdir(parents=True, exist_ok=True)


config = AutoConfig.from_pretrained(model_config)
config.save_pretrained(f"{model_dir}")

raw_dataset = load_dataset("oscar", f"unshuffled_deduplicated_{language}")

tokenizer = ByteLevelBPETokenizer()


def batch_iterator(batch_size=1000):
    for i in range(0, len(raw_dataset), batch_size):
        yield raw_dataset["train"][i: i + batch_size]["text"]


tokenizer.train_from_iterator(batch_iterator(), vocab_size=config.vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])


tokenizer.save(f"{model_dir}/tokenizer.json")

# tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
# max_seq_length = 1024
max_seq_length = int(args[1])


raw_dataset["train"] = load_dataset(
    "oscar", f"unshuffled_deduplicated_{language}", split="train[5%:]")
raw_dataset["validation"] = load_dataset(
    "oscar", f"unshuffled_deduplicated_{language}", split="train[:5%]")

# these cells should be commented out to run on full dataset

raw_dataset["train"] = raw_dataset["train"].select(range(10000))
raw_dataset["validation"] = raw_dataset["validation"].select(range(1000))


tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = raw_dataset.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=raw_dataset["train"].column_names)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i: i + max_seq_length]
            for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_datasets = tokenized_datasets.map(
    group_texts, batched=True, num_proc=4)


per_device_batch_size = int(args[2])
num_epochs = 3
training_seed = 0
learning_rate = 3e-4

# total_batch_size = 16
# print("total batch size: {}".format(total_batch_size))

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForMaskedLM.from_pretrained(model_config)
model = torch.compile(model)
# model = BertModel(config)


training_args = TrainingArguments(
    output_dir="tmp_trainer",
    # evaluation_strategy="epoch",
    weight_decay=0.01,
    # push_to_hub=True,
    # bf16=True,
    fp16=True,
    do_eval = False,
    per_device_train_batch_size=per_device_batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    seed=training_seed,
    push_to_hub=False,
    adam_beta2=0.98,
    remove_unused_columns=False,
)


class TimeHistory(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epoch_time_start = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_time_start = time.time()
        # print("!!{}".format(self.epoch_time_start))
        # print("epoch begin, time: {}".format(time.time()-self.epoch_time_start))

    def on_epoch_end(self, args, state, control, **kwargs):
        # self.times.append(time.time() - self.epoch_time_start)
        print("epoch time: {}".format(time.time()-self.epoch_time_start))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    callbacks = [TimeHistory()]
)

trainer.train()

