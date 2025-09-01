import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist


class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
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
    Block-aware denoising: sample ONE future block and corrupt only that block.
    Mirrors a single block update at inference time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = kwargs["tokenizer"]
        self.mask_token_id = kwargs.get("mask_token_id", self.tokenizer.mask_token_id)
        assert self.mask_token_id is not None, "Provide a mask_token_id or tokenizer.mask_token_id must be set."

        self.max_length = kwargs.get("max_length", None)
        # New knobs to mirror inference:
        self.block_length = kwargs.get("block_length", 32)
        self.gen_length   = kwargs.get("gen_length", 128)  # region after the prompt we treat as "future"
        self.eps          = kwargs.get("eps", 1e-3)

    def _sample_block_bounds(self, prompt_len, seq_len):
        # future window where blocks live
        start_future = prompt_len
        end_future   = min(prompt_len + self.gen_length, seq_len)
        if end_future - start_future < self.block_length:
            # fallback: shrink block if sequence is short
            blk_len = max(1, end_future - start_future)
            blk_start = start_future
            blk_end   = end_future
            return blk_start, blk_end, blk_len

        # uniform sample of a block index inside the future window
        max_start = end_future - self.block_length
        blk_start = torch.randint(low=start_future, high=max_start + 1, size=(1,)).item()
        blk_end   = blk_start + self.block_length
        return blk_start, blk_end, self.block_length

    def forward_process_block(self, input_ids, prompt_lengths):
        """
        Only corrupt the sampled block. Corruption rate ~ Bernoulli(t) with per-example t \in (eps, 1-eps).
        """
        B, N = input_ids.shape
        noisy = input_ids.clone()
        labels = input_ids.clone()

        # Per-example t (scalar), broadcast to chosen block later
        t = torch.rand((B,), device=input_ids.device)
        t = (1 - self.eps) * t + self.eps  # avoid degenerate 0/1

        # Build masks
        t_full = torch.zeros((B, N), device=input_ids.device, dtype=input_ids.dtype)  # will store per-token t
        mask_indices = torch.zeros((B, N), device=input_ids.device, dtype=torch.bool)

        idxs = torch.arange(N, device=input_ids.device).unsqueeze(0)  # [1, N]
        for b in range(B):
            prompt_len = int(prompt_lengths[b].item())
            blk_start, blk_end, blk_len = self._sample_block_bounds(prompt_len, N)

            # Bernoulli(t_b) only inside the block
            m = torch.rand((blk_len,), device=input_ids.device) < t[b]
            mask_indices[b, blk_start:blk_end] = m
            t_full[b, blk_start:blk_end] = (t[b] * torch.ones((blk_len,), device=input_ids.device)).to(t_full.dtype)

            # NEVER corrupt the prompt region
            mask_prompt = (idxs < prompt_len)
            mask_indices[b] = torch.where(mask_prompt[0], torch.zeros_like(mask_indices[b]), mask_indices[b])

        # Apply corruption on the selected positions only
        noisy[mask_indices] = self.mask_token_id

        # Labels: compute loss only where we masked
        labels[~mask_indices] = -100

        # Also, mask out loss for the prompt explicitly
        for b in range(B):
            prompt_len = int(prompt_lengths[b].item())
            labels[b, :prompt_len] = -100

        return noisy, t_full, mask_indices, labels

    def __call__(self, batch):
        batch = super().__call__(batch)
        input_ids = batch["input_ids"]
        prompt_lengths = batch.get("prompt_lengths", None)
        assert prompt_lengths is not None, "Batch must include 'prompt_lengths' to place blocks after prompt."

        noisy_batch, t_full, mask_indices, labels = self.forward_process_block(input_ids, prompt_lengths)

        # package
        batch["labels"] = labels.long()
        batch["input_ids"] = noisy_batch.long()
        batch["t"] = t_full  # per-token t; unmasked tokens may be zeros (ignored by labels=-100 anyway)
        batch["num_prompt_tokens"] = prompt_lengths.sum() if isinstance(prompt_lengths, torch.Tensor) else 0
        return batch

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""


# def preprocess_dataset(data, tokenizer, max_length, test_split=0.01):
#     preprocessed_data = []
#     for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
#         question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
#         trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"
#         prompt = [{"role": "user", "content": question}]
#         response = [{"role": "assistant", "content": trajectory}]
#         inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
#         prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
#         tokenized_input = tokenizer(
#             inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
#         ).input_ids.squeeze(0)
#         num_tokens = tokenized_input.shape[0]
#         tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
#         preprocessed_data.append(
#             {
#                 "input_ids": tokenized_input,
#                 "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
#             }
#         )

#     random.shuffle(preprocessed_data)
#     test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
#     train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
#     return train_data, test_data

def preprocess_dataset(data, tokenizer, max_length, test_split=0.01, with_reasoning=True):
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        if with_reasoning:
            trajectory = f"<reasoning>{data[i]['reasoning']}</reasoning>\n<answer>{data[i]['solution']}</answer>"
        else:
            trajectory = f"<reasoning>\n</reasoning>\n<answer>{data[i]['solution']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    # random.shuffle(preprocessed_data)
    print(int(len(preprocessed_data) * test_split))
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data