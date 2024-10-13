import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


MODEL_NAME=  "mistralai/Mistral-7B-v0.1"

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_double_quant=False,
    )

model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=bnb_config, 
        trust_remote_code=True,
    )


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


model.config.pretraining_tp = 1 
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    target_modules=find_all_linear_names(model),
    bias="none",
    task_type="CAUSAL_LM", 
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    output_dir="./checkpoints/mistral",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,
    bf16=False,
    max_grad_norm=0.1,
    num_train_epochs=1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    save_total_limit=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

model.config.use_cache = False

train_dataset = load_dataset("fwnlp/data_advisor_safety_alignment", split="train")
eval_dataset = load_dataset("fwnlp/data_advisor_safety_alignment", split="validation")
helpful_dataset = load_dataset("json", data_files="data/alpagasus.jsonl", split="train")

train_dataset = concatenate_datasets([train_dataset, helpful_dataset])

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"You are a helpful assistant. USER: {example['prompt'][i]} ASSISTANT: {example['response'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "ASSISTANT:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
