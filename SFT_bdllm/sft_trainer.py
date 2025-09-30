import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
import json

class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        # print((labels != -100).sum())
        # print(inputs["input_ids"].shape)
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        num_of_tokens = inputs["input_ids"].numel()
        num_of_labels = labels.numel()
        num_of_non_masked_tokens = (labels != -100).sum()
        loss = loss.sum() / ((labels != -100).sum())
        return loss if not return_outputs else (loss, outputs)


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out

class dLLMDataCollator(DefaultDataCollator):
    """
    Data collator for block diffusion training.
    Only applies diffusion to the current block, keeping prompt and previous blocks unchanged.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process_block(self, batch, eps=1e-3):
        """
        Apply forward diffusion process only to the current block.
        """
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        
        # Get time steps
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]
        
        t = (1 - eps) * t + eps
        t_expanded = t[:, None].repeat(1, N)
        
        # Create mask for the current block only
        current_block_starts = batch["current_block_start"]
        current_block_ends = batch["current_block_end"]
        
        # Initialize mask indices (False = no masking)
        mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Apply masking only to current blocks
        for i in range(B):
            block_start = current_block_starts[i] if isinstance(current_block_starts[i], int) else current_block_starts[i].item()
            block_end = current_block_ends[i] if isinstance(current_block_ends[i], int) else current_block_ends[i].item()
            
            if block_start < block_end and block_end <= N:
                # Generate random mask for current block
                block_length = block_end - block_start
                block_mask = torch.rand(block_length, device=input_ids.device) < t[i]
                while not block_mask.any():
                    block_mask = torch.rand(block_length, device=input_ids.device) < t[i]
                mask_indices[i, block_start:block_end] = block_mask
        
        # Apply masking
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        
        return noisy_batch, t_expanded, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        
        # Apply block diffusion
        noisy_batch, batch["t"], mask_indices = self.forward_process_block(batch)
        
        # Initialize labels to -100 (no loss) for all positions
        batch["labels"][:] = -100
        
        # Only compute loss on tokens within the current block
        current_block_starts = batch["current_block_start"]
        current_block_ends = batch["current_block_end"]
        
        B = batch["input_ids"].shape[0]
        for i in range(B):
            block_start = current_block_starts[i] if isinstance(current_block_starts[i], int) else current_block_starts[i].item()
            block_end = current_block_ends[i] if isinstance(current_block_ends[i], int) else current_block_ends[i].item()
            
            if block_start < block_end:
                # Only set labels for the current block, and only for masked positions
                current_block_mask = mask_indices[i, block_start:block_end]
                current_block_labels = batch["input_ids"][i, block_start:block_end].clone()
                
                # Set labels only for masked tokens in the current block
                batch["labels"][i, block_start:block_end][current_block_mask] = current_block_labels[current_block_mask]
        
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch["prompt_lengths"]
            if prompt_lengths.dim() > 1:
                prompt_lengths = prompt_lengths.squeeze(-1)
            for prompt_len in prompt_lengths:
                batch["num_prompt_tokens"] += prompt_len.item()
        
        batch["input_ids"] = noisy_batch.long()

        # Clean up temporary keys that aren't needed for training
        keys_to_remove = ["current_block_start", "current_block_end", "block_idx", "total_blocks", "prompt_lengths"]
        for key in keys_to_remove:
            if key in batch:
                batch.pop(key)
        
        return batch

def preprocess_dataset(data, tokenizer, max_length, block_size=32, test_split=0.01, with_reasoning=True):
    """
    Preprocess dataset for block diffusion training.
    Each data point is split into multiple training examples, one for each block.
    
    Args:
        data: Raw dataset
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        block_size: Size of each block for diffusion
        test_split: Fraction of data for testing
        with_reasoning: Whether to include reasoning in the response
    
    Returns:
        train_data, test_data: Preprocessed datasets
    """
    preprocessed_data = []
    # raw_split_idx = int(len(data) * (1 - test_split))
    raw_split_idx = 128
    test_split_idx = -1
    for i in tqdm(range(len(data)), desc="Preprocessing dataset for block diffusion"):
        if i == raw_split_idx and test_split_idx == -1:
            test_split_idx = len(preprocessed_data)
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        
        if with_reasoning:
            trajectory = f"<reasoning>{data[i]['reasoning']}</reasoning>\n<answer>{data[i]['solution']}</answer>"
        else:
            trajectory = f"<reasoning>\n</reasoning>\n<answer>{data[i]['solution']}</answer>"
        
        # Create the full conversation
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        
        # Get the full message
        message_text = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        parts = message_text.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) != 2:
            message_text = "<|start_header_id|>assistant<|end_header_id|>".join(parts[:-1])
        tokenized_message = tokenizer(message_text, return_tensors="pt", max_length=max_length, padding="max_length").input_ids.squeeze(0)
        if tokenized_message.shape[0] > max_length:
            continue
        response_start_idx = message_text.find(trajectory)
        # Get the prompt part
        prompt_text = message_text[:response_start_idx]
        tokenized_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = tokenized_prompt.input_ids.squeeze(0)
        
        # Get the response part
        response_tokens = tokenized_message[len(prompt_tokens):]
        # response_text = message_text[response_start_idx:]
        # tokenized_response = tokenizer(response_text, return_tensors="pt")
        # response_tokens = tokenized_response.input_ids.squeeze(0)

        
        # Find the actual prompt length in the tokenized message
        # by matching the prompt tokens with the beginning of the tokenized message
        prompt_length = len(prompt_tokens)
        
        # Split response into blocks
        num_blocks = (len(response_tokens) + block_size - 1) // block_size  # Ceiling division
        
        for block_idx in range(num_blocks):
            # Calculate block boundaries in response tokens
            block_start = block_idx * block_size
            block_end = min((block_idx + 1) * block_size, len(response_tokens))
            
            # Calculate boundaries in the full tokenized message
            # Structure: [prompt] + [previous blocks] + [current block] + [future blocks] + [padding]
            message_prompt_end = prompt_length
            message_prev_blocks_end = message_prompt_end + block_start
            message_current_block_end = message_prompt_end + block_end
            
            # Create the input sequence by modifying the original tokenized message
            input_sequence = tokenized_message.clone()
            
            # Get mask token ID
            mask_token_id = tokenizer.mask_token_id
            if mask_token_id is None:
                mask_token_id = 126336
            
            # Replace future blocks (after current block) with mask tokens
            # This includes both future response tokens AND padding tokens
            # The model needs to learn to predict both content and when to stop (padding)
            if message_current_block_end < max_length:
                future_start = message_current_block_end
                input_sequence[future_start:] = mask_token_id
            
            # Calculate current block boundaries in the final sequence
            current_block_start = message_prev_blocks_end
            current_block_end = min(message_current_block_end, max_length)
            
            # Skip if current block is completely truncated
            if current_block_start >= max_length:
                continue
                
            preprocessed_data.append({
                "input_ids": input_sequence,
                "prompt_lengths": torch.tensor([prompt_length]),
                "current_block_start": current_block_start,
                "current_block_end": current_block_end,
                "block_idx": block_idx,
                "total_blocks": num_blocks,
            })
            # print(f"block_start: {preprocessed_data[-1]['current_block_start']}")
            # print(f"block_end: {preprocessed_data[-1]['current_block_end']}")
    # Split into train and test using the recorded split index
    if test_split_idx == -1:
        test_split_idx = len(preprocessed_data)
        
    train_data = preprocessed_data[:test_split_idx]
    test_data = preprocessed_data[test_split_idx:]

    return train_data, test_data

def to_jsonable(x):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        return x.tolist()
    return x