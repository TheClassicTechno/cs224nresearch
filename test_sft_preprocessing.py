"""Test script to verify SFT preprocessing behavior."""

import os
from transformers import AutoTokenizer
import torch
print(torch.__version__)

# Test with a small example
MODEL_ID = "Qwen/Qwen3-1.7B"

print("=" * 60)
print("Testing SFT Preprocessing Behavior")
print("=" * 60)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Example data
example = {
    "prompt": "What are the symptoms of diabetes?",
    "response": "Common symptoms include increased thirst, frequent urination, and fatigue."
}

print(f"\nExample prompt: {example['prompt']}")
print(f"Example response: {example['response']}")

# ── CURRENT (BUGGY) APPROACH ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CURRENT APPROACH (Line 54): prompt + response concatenation")
print("=" * 60)

text_buggy = example["prompt"] + example["response"]
enc_buggy = tokenizer(
    text_buggy,
    truncation=True,
    max_length=1024,
    padding="max_length",
)
labels_buggy = enc_buggy["input_ids"].copy()

print(f"\nFull text: {text_buggy}")
print(f"\nTokenized length: {len(enc_buggy['input_ids'])}")
print(f"Labels length: {len(labels_buggy)}")
print(f"\n⚠️  PROBLEM: ALL tokens (including prompt) have labels!")
print(f"   This means model learns to predict the prompt too.")

# Decode to show what's being trained
decoded_prompt = tokenizer.decode(enc_buggy["input_ids"][:20], skip_special_tokens=False)
print(f"\nFirst 20 tokens (prompt part): {decoded_prompt}")
print(f"   These tokens WILL be trained on (wrong!)")

# ── CORRECT APPROACH (Chat Template) ─────────────────────────────────
print("\n" + "=" * 60)
print("CORRECT APPROACH: Chat template with label masking")
print("=" * 60)

messages = [
    {"role": "user", "content": example["prompt"]},
    {"role": "assistant", "content": example["response"]}
]

# Format with chat template
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
print(f"\nFormatted with chat template:\n{formatted}")

# Tokenize
enc_correct = tokenizer(
    formatted,
    truncation=True,
    max_length=1024,
    padding="max_length",
    return_tensors="pt",
)

# Create labels: mask prompt tokens with -100
labels_correct = enc_correct["input_ids"].clone()
# Find where assistant response starts (after user message)
# This is approximate - in practice you'd find the exact token position
user_tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": example["prompt"]}],
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
)
prompt_len = user_tokens.shape[1]
labels_correct[:, :prompt_len] = -100  # Mask prompt tokens

print(f"\nTokenized length: {enc_correct['input_ids'].shape[1]}")
print(f"Prompt tokens (masked): {prompt_len}")
print(f"Response tokens (trained): {enc_correct['input_ids'].shape[1] - prompt_len}")
print(f"\n✓ CORRECT: Only response tokens have labels (prompt masked with -100)")

# Show the difference
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Current approach trains on: {len([l for l in labels_buggy if l != tokenizer.pad_token_id])} tokens (WRONG)")
print(f"Correct approach trains on: {len([l for l in labels_correct[0] if l != -100 and l != tokenizer.pad_token_id])} tokens (CORRECT)")
print("\n⚠️  The current code will make the model learn to echo the prompt!")
