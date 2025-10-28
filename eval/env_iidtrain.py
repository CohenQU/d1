import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
from generate import generate
import random
import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, is_equiv

AIME2024_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""

def extract_last_boxed(solution: str):
    """Extract the last \boxed{...} from solution. Return None if not found."""
    matches = re.findall(r'\\boxed\{(.*?)\}', solution, re.DOTALL)
    return matches[-1].strip() if matches else None

def process(example):
    answer = extract_last_boxed(example["solution"])
    return {
        "problem": example["question"],
        "answer": answer
    }


class IIDTrainDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=AIME2024_SYSTEM_PROMPT,
        dataset_start=0,
        dataset_end=None,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, dataset_start, dataset_end)

    def load_test_dataset(self):
        self.dataset = load_dataset("CohenQu/Mixture-of-Thoughts-math-4k-80", split="train")
        self.dataset = self.dataset.select(range(50))
        self.dataset = self.dataset.map(process)
        self.dataset = self.dataset.filter(lambda ex: ex["answer"] is not None)

    def load_few_shot_examples(self):
        train_data = load_dataset("EleutherAI/hendrycks_math", ("algebra"), split="train")
        few_shot_examples = []
        samples = random.sample(range(len(train_data)), self.num_examples)
        for example in samples:
            few_shot_examples.append(
                {"question": train_data[example]["problem"], "answer": train_data[example]["solution"]}
            )
        return few_shot_examples

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
