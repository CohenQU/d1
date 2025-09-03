#!/usr/bin/env python3
"""
Builds a filtered math split from `open-r1/Mixture-of-Thoughts` and pushes it to
`CohenQu/Mixture-of-Thoughts-4k` with config (subset) = "math" and split = "train".

Conditions kept:
- example["num_tokens"] < 4000
- len(example["messages"]) == 2
- "<think>" and "</think>" both appear in example["messages"][1]["content"]

Outputs columns:
- question
- reasoning  (content inside <think>...</think>)
- solution   (content after </think>)
"""

from datasets import load_dataset
from huggingface_hub import login
import argparse
import matplotlib.pyplot as plt


def has_think_tags(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return ("<think>" in text) and ("</think>" in text)


def extract_reasoning_and_solution(text: str):
    try:
        after_open = text.split("<think>", 1)[1]
        reasoning, after_close = after_open.split("</think>", 1)
        reasoning = reasoning.strip()
        solution = after_close
        return reasoning, solution
    except Exception:
        return None, None


def keep_example(ex):
    if "num_tokens" not in ex:
        return False
    try:
        if int(ex["num_tokens"]) >= 4000:
            return False
    except Exception:
        return False

    msgs = ex.get("messages", None)
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False

    user_msg = msgs[0] if isinstance(msgs[0], dict) else None
    assistant_msg = msgs[1] if isinstance(msgs[1], dict) else None
    if user_msg is None or assistant_msg is None:
        return False

    content0 = user_msg.get("content", None)
    content1 = assistant_msg.get("content", None)

    if not isinstance(content0, str) or not isinstance(content1, str):
        return False

    if not has_think_tags(content1):
        return False

    return True


def transform_example(ex):
    msgs = ex["messages"]
    question = msgs[0]["content"].strip()
    content1 = msgs[1]["content"].strip()

    reasoning, solution = extract_reasoning_and_solution(content1)
    if reasoning is None:
        return {"question": None, "reasoning": None, "solution": None}

    return {
        "question": question,
        "reasoning": reasoning,
        "solution": solution,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="CohenQu/Mixture-of-Thoughts-4k-all",
                        help="Target repo to push to.")
    parser.add_argument("--config_name", type=str, default="all",
                        help="Config (subset) name to use for the pushed dataset.")
    parser.add_argument("--private", action="store_true",
                        help="Push to a private repo.")
    args = parser.parse_args()

    print("Loading source dataset: open-r1/Mixture-of-Thoughts (config='all', split='train')...")
    ds = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="train")

    print("Filtering examples...")
    ds_filtered = ds.filter(keep_example, num_proc=None)

    # --- NEW: plot num_tokens distribution ---
    print("Plotting num_tokens distribution...")
    num_tokens_list = [int(ex["num_tokens"]) for ex in ds_filtered]
    plt.hist(num_tokens_list, bins=50, color="steelblue", edgecolor="black")
    plt.xlabel("num_tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of num_tokens (<4000)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("/home/yuxiaoq/projects/flexible-ordering-dllm/scripts/num_tokens_distribution.png")
    print("Saved histogram to num_tokens_distribution.png")

    # print("Transforming to {question, reasoning, solution}...")
    # ds_mapped = ds_filtered.map(
    #     transform_example,
    #     remove_columns=ds_filtered.column_names,
    #     desc="Extract QA fields",
    # )

    # ds_final = ds_mapped.filter(
    #     lambda ex: ex["question"] is not None and ex["reasoning"] is not None and ex["solution"] is not None
    # )

    # ds_final = ds_final.shuffle(seed=42)

    # print(f"Final dataset size: {len(ds_final)} rows")

    # print(f"Pushing to hub: repo_id={args.repo_id}, config_name={args.config_name}, split='train'...")
    # ds_final.push_to_hub(
    #     repo_id=args.repo_id,
    #     config_name=args.config_name,
    #     split="train",
    #     private=args.private,
    # )

    print("Done!")


if __name__ == "__main__":
    main()
