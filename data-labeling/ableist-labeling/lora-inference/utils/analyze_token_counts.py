#!/usr/bin/env python3
"""
Quick script to analyze token counts in training data prompts.
"""

import json
import os
from transformers import AutoTokenizer
import numpy as np

def analyze_token_counts(data_path, model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Analyze token counts in the training data."""
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Analyze token counts
    token_counts = []
    prompt_lengths = []
    
    for i, item in enumerate(data):
        # Format prompt (same as training)
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        
        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        token_count = len(tokens)
        token_counts.append(token_count)
        prompt_lengths.append(len(prompt))
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(data)} samples...")
    
    # Calculate statistics
    token_counts = np.array(token_counts)
    prompt_lengths = np.array(prompt_lengths)
    
    print("\n" + "="*60)
    print("TOKEN COUNT ANALYSIS")
    print("="*60)
    print(f"Total samples: {len(data)}")
    print(f"Token count statistics:")
    print(f"  Mean: {np.mean(token_counts):.1f}")
    print(f"  Median: {np.median(token_counts):.1f}")
    print(f"  Min: {np.min(token_counts)}")
    print(f"  Max: {np.max(token_counts)}")
    print(f"  Std: {np.std(token_counts):.1f}")
    print(f"  25th percentile: {np.percentile(token_counts, 25):.1f}")
    print(f"  75th percentile: {np.percentile(token_counts, 75):.1f}")
    print(f"  90th percentile: {np.percentile(token_counts, 90):.1f}")
    print(f"  95th percentile: {np.percentile(token_counts, 95):.1f}")
    print(f"  99th percentile: {np.percentile(token_counts, 99):.1f}")
    
    print(f"\nCharacter count statistics:")
    print(f"  Mean: {np.mean(prompt_lengths):.1f}")
    print(f"  Median: {np.median(prompt_lengths):.1f}")
    print(f"  Min: {np.min(prompt_lengths)}")
    print(f"  Max: {np.max(prompt_lengths)}")
    
    # Check how many samples exceed different token limits
    print(f"\nToken limit analysis:")
    limits = [512, 1024, 1536, 2048, 2560, 3072]
    for limit in limits:
        over_limit = np.sum(token_counts > limit)
        percentage = (over_limit / len(token_counts)) * 100
        print(f"  Samples > {limit} tokens: {over_limit} ({percentage:.1f}%)")
    
    # Find the optimal max_length
    print(f"\nRecommended max_length settings:")
    for percentile in [90, 95, 99]:
        recommended = int(np.percentile(token_counts, percentile))
        over_limit = np.sum(token_counts > recommended)
        percentage = (over_limit / len(token_counts)) * 100
        print(f"  {percentile}th percentile ({recommended} tokens): {over_limit} samples ({percentage:.1f}%) would be truncated")
    
    # Show some examples of long prompts
    print(f"\nLongest prompts (top 5):")
    sorted_indices = np.argsort(token_counts)[::-1]
    for i in range(min(5, len(data))):
        idx = sorted_indices[i]
        print(f"  Sample {idx}: {token_counts[idx]} tokens, {prompt_lengths[idx]} chars")
        print(f"    Instruction: {data[idx]['instruction'][:100]}...")
        print(f"    Input: {data[idx]['input'][:100]}...")
        print()
    
    return token_counts, prompt_lengths

if __name__ == "__main__":
    data_path = "/n/fs/peas-lab/hayoung/chast-abl/script_prep/data/my_train.json"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        exit(1)
    
    token_counts, prompt_lengths = analyze_token_counts(data_path)
    
    print("\nAnalysis complete!")
