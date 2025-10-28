import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import os
from sft_trainer import *
import torch.distributed as dist
import random
import numpy as np
from transformers.trainer_utils import get_last_checkpoint

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", padding_side="right", trust_remote_code=True, use_fast=True
    )

question = "What is the capital of France?"
reasoning = "The capital of France is Paris."
answer = "Paris"
trajectory = f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
prompt = [{"role": "user", "content": question}]
response = [{"role": "assistant", "content": trajectory}]

message_text = tokenizer.apply_chat_template(prompt + response, tokenize=False)
message_tokens = tokenizer(message_text, return_tensors="pt", max_length=128, padding="max_length").input_ids.squeeze(0)
print(message_text)
print(message_tokens)

# find where the trajectory starts
response_start_idx = message_text.find(trajectory)
prompt_text = message_text[:response_start_idx]
response_text = message_text[response_start_idx:]

# prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False)
prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
print(prompt_text)
print(prompt_tokens)

# response_text = tokenizer.apply_chat_template(response, tokenize=False, add_generation_prompt=False, add_special_tokens=False)
print(response_text)
response_tokens = tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
print(response_tokens)

# print(tokenizer)


# inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False)
# print(inputs)

# inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=True)
# print(inputs)


# inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False, add_special_tokens=False)
# print(inputs)

# inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False, add_special_tokens=True)
# tokenized_inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=False)
# print(tokenized_inputs)
# tokenized_inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)
# print(tokenized_inputs)

# print(tokenizer)
