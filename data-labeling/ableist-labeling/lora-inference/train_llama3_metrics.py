#!/usr/bin/env python3
"""
Modular script for fine-tuning Llama3.1-8b using LoRA and PEFT for metrics detection.
Supports configurable parameters for training hyperparameters and LoRA settings.

Must be in the format:
### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}
"""

import argparse
import json
import yaml
import os
import sys
import torch
import logging
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
from datasets import Dataset
import wandb
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import re
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create a more robust logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Use stdout explicitly
        ],
        force=True  # Force reconfiguration
    )
    
    # Disable wandb's verbose logging to prevent conflicts
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")


class MetricsDataset:
    """Dataset class for loading and preprocessing metrics detection data."""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format the prompt for training."""
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        return prompt
    
    def tokenize_function(self, example):
        """Tokenize a single example for training."""
        # Create the instruction + input part (what the model sees as context)
        instruction_input = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        
        # Create the full prompt (context + response)
        full_prompt = instruction_input + example['output']
        
        # Tokenize the full prompt
        full_tokenized = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Tokenize just the instruction + input part to find where response starts
        instruction_tokenized = self.tokenizer(
            instruction_input,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Create labels: mask instruction/input part with -100, keep response part
        labels = full_tokenized["input_ids"].copy()
        instruction_length = len(instruction_tokenized["input_ids"])
        
        # Mask the instruction + input part (set to -100 so it's ignored in loss)
        for i in range(min(instruction_length, len(labels))):
            labels[i] = -100
        
        result = {
            "input_ids": full_tokenized["input_ids"],
            "attention_mask": full_tokenized["attention_mask"],
            "labels": labels
        }
        
        return result
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training."""
        data = self.load_data(data_path)
        
        # Convert to the format expected by the tokenizer
        formatted_data = {
            'instruction': [item['instruction'] for item in data],
            'input': [item['input'] for item in data],
            'output': [item['output'] for item in data]
        }
        
        dataset = Dataset.from_dict(formatted_data)
        
        # Process one example at a time to avoid tensor shape issues
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=False,  # Process one at a time
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset


class MetricsEvaluator:
    """Custom evaluator for computing metrics during training."""
    
    def __init__(self, tokenizer, model, metric_names):
        self.tokenizer = tokenizer
        self.model = model
        self.metric_names = metric_names
        
        # Validate that model and tokenizer are compatible
        logger.info(f"MetricsEvaluator initialized with {len(metric_names)} metrics: {metric_names}")
        
    def parse_metrics_output(self, output: str) -> Dict[str, int]:
        """Parse the metrics output from the model response (YAML format)."""
        try:
            # First try to find YAML in code blocks
            yaml_start = output.find('```yaml')
            if yaml_start != -1:
                yaml_start += 7  # Skip '```yaml'
                yaml_end = output.find('```', yaml_start)
                if yaml_end != -1:
                    yaml_str = output[yaml_start:yaml_end].strip()
                    # Clean the YAML string
                    yaml_str = self._clean_yaml_string(yaml_str)
                    metrics = yaml.safe_load(yaml_str)
                    if isinstance(metrics, dict):
                        # Ensure all expected metrics are present
                        for metric in self.metric_names:
                            if metric not in metrics:
                                metrics[metric] = 0
                        return metrics
            
            # Try to extract YAML-like content from the beginning of the response
            lines = output.strip().split('\n')
            yaml_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Stop if we hit explanatory text (lines without colons)
                if ':' not in line and len(yaml_lines) > 0:
                    break
                # Check if line matches metric pattern
                if any(metric in line for metric in self.metric_names):
                    # Clean the line and extract metric: value
                    cleaned_line = self._clean_metric_line(line)
                    if cleaned_line:
                        yaml_lines.append(cleaned_line)
            
            if yaml_lines:
                yaml_str = '\n'.join(yaml_lines)
                metrics = yaml.safe_load(yaml_str)
                if isinstance(metrics, dict):
                    # Ensure all expected metrics are present
                    for metric in self.metric_names:
                        if metric not in metrics:
                            metrics[metric] = 0
                    return metrics
            
            # Fallback: regex-based extraction
            metrics = {metric: 0 for metric in self.metric_names}
            for metric in self.metric_names:
                # Look for patterns like "MetricName: 1" or "MetricName: 0"
                import re
                pattern = rf"{re.escape(metric)}\s*:\s*(\d+)"
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    metrics[metric] = int(match.group(1))
            return metrics
            
        except (yaml.YAMLError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse YAML output: {e}")
            # If YAML parsing fails, return default values
            return {metric: 0 for metric in self.metric_names}
    
    def _clean_yaml_string(self, yaml_str: str) -> str:
        """Clean YAML string by removing markdown formatting and invalid syntax."""
        lines = yaml_str.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip lines that start with dashes (bullet points)
            if line.startswith('-'):
                continue
            # Remove markdown bold formatting
            line = line.replace('**', '')
            # Skip lines without colons (explanatory text)
            if ':' not in line:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_metric_line(self, line: str) -> str:
        """Clean a single metric line and return in proper YAML format."""
        import re
        
        # Remove markdown formatting
        line = line.replace('**', '').replace('*', '')
        # Remove bullet points
        line = re.sub(r'^[-•]\s*', '', line)
        
        # Try to extract metric name and value
        for metric in self.metric_names:
            if metric in line:
                # Look for the value (0 or 1)
                value_match = re.search(r':\s*([01])', line)
                if value_match:
                    return f"{metric}: {value_match.group(1)}"
        
        return None
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from the model."""
        try:
            # Clear CUDA cache before generation to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Use 2048 max_length consistently
            ).to(self.model.device)
            
            # Validate input token IDs are within vocabulary range
            vocab_size = len(self.tokenizer)
            input_ids = inputs['input_ids']
            max_token_id = torch.max(input_ids).item()
            min_token_id = torch.min(input_ids).item()
            
            if max_token_id >= vocab_size or min_token_id < 0:
                logger.error(f"Invalid token IDs detected: min={min_token_id}, max={max_token_id}, vocab_size={vocab_size}")
                return ""
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,  # Use pad_token_id instead of eos_token_id
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                    # attention_mask is already included in **inputs, no need to pass explicitly
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return ""
    
    def generate_batch_responses(self, prompts: List[str], max_new_tokens: int = 512) -> List[str]:
        """Generate responses for a batch of prompts."""
        try:
            # Clear CUDA cache before batch generation to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Tokenize all prompts
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Use 2048 max_length consistently
                padding=True
            ).to(self.model.device)
            
            # Validate input token IDs are within vocabulary range
            vocab_size = len(self.tokenizer)
            input_ids = inputs['input_ids']
            max_token_id = torch.max(input_ids).item()
            min_token_id = torch.min(input_ids).item()
            
            if max_token_id >= vocab_size or min_token_id < 0:
                logger.error(f"Invalid token IDs detected in batch: min={min_token_id}, max={max_token_id}, vocab_size={vocab_size}")
                # Return empty responses for the batch
                return [""] * len(prompts)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,  # Use pad_token_id instead of eos_token_id
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                    # attention_mask is already included in **inputs, no need to pass explicitly
                )
            
            # Decode responses for each prompt
            responses = []
            for i, prompt in enumerate(prompts):
                try:
                    # Get only the new tokens for this prompt
                    input_length = inputs['input_ids'][i].shape[0]
                    new_tokens = outputs[i][input_length:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    responses.append(response.strip())
                except Exception as e:
                    logger.warning(f"Error decoding response for prompt {i}: {e}")
                    responses.append("")
            
            return responses
            
        except Exception as e:
            logger.error(f"Error in generate_batch_responses: {e}")
            # Return empty responses for all prompts on error
            return [""] * len(prompts)
    
    def evaluate_dataset(self, dataset, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate the model on a dataset."""
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        predictions = []
        ground_truths = []
        sample_examples = []  # Store examples for logging
        
        # Create progress bar for evaluation
        progress_bar = tqdm(
            enumerate(dataset), 
            total=len(dataset),
            desc="Evaluating",
            unit="samples",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Process in batches for better GPU utilization
        batch_size = 16  # Process 16 samples at once
        batch_prompts = []
        batch_items = []
        
        for i, item in progress_bar:
            # Format prompt (same as training)
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            
            batch_prompts.append(prompt)
            batch_items.append(item)
            
            # Process batch when full or at the end
            if len(batch_prompts) == batch_size or i == len(dataset) - 1:
                # Generate responses for the batch
                batch_responses = self.generate_batch_responses(batch_prompts)
                
                # Process each response in the batch
                for j, (response, item) in enumerate(zip(batch_responses, batch_items)):
                    sample_idx = i - len(batch_items) + j + 1
                    
                    # Parse predictions and ground truth
                    pred_metrics = self.parse_metrics_output(response)
                    try:
                        # Parse ground truth YAML
                        ground_truth_output = item.get('output', '')
                        true_metrics = yaml.safe_load(ground_truth_output)
                        if not isinstance(true_metrics, dict):
                            raise ValueError("Ground truth is not a dictionary")
                        # Ensure all expected metrics are present
                        for metric in self.metric_names:
                            if metric not in true_metrics:
                                true_metrics[metric] = 0
                    except (yaml.YAMLError, ValueError, KeyError) as e:
                        logger.warning(f"Failed to parse ground truth YAML for sample {sample_idx}: {e}")
                        # Use default values if parsing fails
                        true_metrics = {metric: 0 for metric in self.metric_names}
                    
                    # Store first 2 examples for logging
                    if len(sample_examples) < 2:
                        sample_examples.append({
                            'sample_idx': sample_idx,
                            'instruction': instruction,
                            'input_text': input_text[:200] + "..." if len(input_text) > 200 else input_text,  # Truncate long inputs
                            'predicted_response': response,  # Truncate long responses
                            'ground_truth': item.get('output', '{}'),
                            'predicted_metrics': pred_metrics,
                            'true_metrics': true_metrics
                        })
                    
                    # Convert to lists for evaluation
                    pred_values = [pred_metrics.get(metric, 0) for metric in self.metric_names]
                    true_values = [true_metrics.get(metric, 0) for metric in self.metric_names]
                    
                    predictions.append(pred_values)
                    ground_truths.append(true_values)
                
                # Clear batch
                batch_prompts = []
                batch_items = []
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        # Overall accuracy
        accuracy = accuracy_score(ground_truths.flatten(), predictions.flatten())
        
        # Per-metric metrics
        metric_results = {}
        for i, metric in enumerate(self.metric_names):
            metric_pred = predictions[:, i]
            metric_true = ground_truths[:, i]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                metric_true, metric_pred, average='binary', zero_division=0
            )
            
            metric_results[metric] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy_score(metric_true, metric_pred)
            }
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            ground_truths.flatten(), predictions.flatten(), average='macro', zero_division=0
        )
        
        results = {
            'overall_accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_metric_results': metric_results,
            'metric_names': self.metric_names,
            'sample_examples': sample_examples
        }
        
        return results


class MetricsCallback(TrainerCallback):
    """Custom callback for evaluation during training."""
    
    def __init__(self, trainer, val_dataset_raw, train_dataset_raw, metric_names, eval_samples=100):
        self.trainer = trainer
        self.val_dataset_raw = val_dataset_raw
        self.train_dataset_raw = train_dataset_raw
        self.metric_names = metric_names
        self.eval_samples = eval_samples
        self.evaluator = None
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.best_val_results = None
        self.evaluation_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Evaluation points
        self.evaluated_points = set()  # Track which points we've already evaluated
        
    def should_evaluate_at_step(self, state) -> bool:
        """Check if we should evaluate at this step based on epoch progress."""
        current_epoch = state.epoch
        for eval_point in self.evaluation_points:
            if abs(current_epoch - eval_point) < 0.05:  # Tighter tolerance
                # Check if we haven't already evaluated this point
                if eval_point not in self.evaluated_points:
                    self.evaluated_points.add(eval_point)
                    return True
        return False
    
    def evaluate_all_datasets(self, state):
        """Evaluate on validation set only. Test set evaluation is disabled."""
        if self.evaluator is None:
            self.evaluator = MetricsEvaluator(
                self.trainer.tokenizer, 
                self.trainer.model, 
                self.metric_names
            )
        
        # Clear CUDA cache before evaluation to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        results = {}
        
        # Evaluate on validation set (full dataset)
        if self.val_dataset_raw is not None:
            logger.info(f"\n=== Evaluating on validation set at epoch {state.epoch:.1f} ===")
            val_results = self.evaluator.evaluate_dataset(
                self.val_dataset_raw, 
                max_samples=None  # Use full dataset
            )
            results['val'] = val_results
            
            # Log validation metrics to wandb
            if wandb.run is not None:
                try:
                    wandb.log({
                        "val/overall_accuracy": val_results['overall_accuracy'],
                        "val/macro_precision": val_results['macro_precision'],
                        "val/macro_recall": val_results['macro_recall'],
                        "val/macro_f1": val_results['macro_f1'],
                        "epoch": state.epoch
                    })
                    
                    # Log per-metric validation results
                    for metric, metric_results in val_results['per_metric_results'].items():
                        clean_metric = metric.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                        wandb.log({
                            f"val/{clean_metric}_precision": metric_results['precision'],
                            f"val/{clean_metric}_recall": metric_results['recall'],
                            f"val/{clean_metric}_f1": metric_results['f1'],
                            f"val/{clean_metric}_accuracy": metric_results['accuracy'],
                            "epoch": state.epoch
                        })
                except Exception as e:
                    logger.warning(f"Failed to log validation metrics to wandb: {e}")
            
            logger.info(f"Validation - Accuracy: {val_results['overall_accuracy']:.4f}, "
                       f"Macro F1: {val_results['macro_f1']:.4f}")
            
            # Log per-metric F1 scores
            logger.info("\n📊 VALIDATION F1 SCORES PER METRIC:")
            for metric, results in val_results['per_metric_results'].items():
                logger.info(f"  {metric}: {results['f1']:.4f}")
            logger.info(f"  Overall Macro F1: {val_results['macro_f1']:.4f}")
            
            # Log sample examples from validation
            if 'sample_examples' in val_results and val_results['sample_examples']:
                logger.info("\n📋 VALIDATION SAMPLE EXAMPLES:")
                for i, example in enumerate(val_results['sample_examples'], 1):
                    logger.info(f"\n--- Example {i} (Sample #{example['sample_idx']}) ---")
                    logger.info(f"Instruction: {example['instruction']}")
                    logger.info(f"Input: {example['input_text']}")
                    logger.info(f"Predicted Response: {example['predicted_response']}")
                    logger.info(f"Ground Truth: {example['ground_truth']}")
                    logger.info(f"Predicted Metrics: {example['predicted_metrics']}")
                    logger.info("---")
            
            # Check if this is the best validation performance
            if val_results['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_results['macro_f1']
                self.best_epoch = state.epoch
                self.best_val_results = val_results
                
                logger.info(f"🎉 NEW BEST VALIDATION F1: {self.best_val_f1:.4f} at epoch {state.epoch:.1f}")
                
                # Only save adapter after actual training has started (not during initial evaluation)
                if state.epoch > 0.1:  # Only save after some training has occurred
                    try:
                        best_adapter_path = os.path.join(self.trainer.args.output_dir, "best_adapter")
                        self.trainer.model.save_pretrained(best_adapter_path)
                        logger.info(f"💾 Best adapter saved to: {best_adapter_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save best adapter: {e}")
                else:
                    logger.info("💾 Skipping adapter save during initial evaluation")
                
                # Log best performance to wandb
                if wandb.run is not None:
                    try:
                        wandb.log({
                            "best_val_f1": self.best_val_f1,
                            "best_epoch": self.best_epoch,
                            "epoch": state.epoch
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log best validation F1 to wandb: {e}")
            else:
                logger.info(f"📊 Validation F1: {val_results['macro_f1']:.4f} (Best: {self.best_val_f1:.4f} at epoch {self.best_epoch:.1f})")
        
        return results
        
    def on_log(self, args, state, control, **kwargs):
        """Log training loss only."""
        # Only log training loss - validation metrics are logged in evaluate_all_datasets
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Evaluate at the beginning of training."""
        logger.info("🚀 Starting training - running initial evaluation...")
        self.evaluate_all_datasets(state)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Evaluate at specific epoch intervals (0.5, 1.0, 1.5, etc.)."""
        # Calculate steps per epoch to determine appropriate check frequency
        total_steps = state.max_steps if state.max_steps > 0 else len(args.train_dataloader) * args.num_train_epochs
        steps_per_epoch = total_steps / args.num_train_epochs if args.num_train_epochs > 0 else 600
        
        # Check every 0.5 epochs (twice per epoch) to catch evaluation points
        check_frequency = max(int(steps_per_epoch * 0.5), 200)  # At least every 200 steps
        
        if state.global_step % check_frequency == 0:
            current_epoch = state.epoch
            
            # Check if we should evaluate at this step
            if self.should_evaluate_at_step(state):
                logger.info(f"\n🔄 Triggering evaluation at epoch {current_epoch:.2f} (step {state.global_step})")
                self.evaluate_all_datasets(state)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate at the end of each epoch."""
        # Check if we should evaluate at this epoch
        if self.should_evaluate_at_step(state):
            self.evaluate_all_datasets(state)
    
    def print_best_results(self):
        """Print the best results on the validation set achieved during training."""
        logger.info("\n" + "="*80)
        logger.info("🏆 BEST RESULTS SUMMARY")
        logger.info("="*80)
        
        if self.best_val_results is not None:
            logger.info(f"Best Validation Performance (Epoch {self.best_epoch}):")
            logger.info(f"  Overall Accuracy: {self.best_val_results['overall_accuracy']:.4f}")
            logger.info(f"  Macro F1: {self.best_val_results['macro_f1']:.4f}")
            logger.info(f"  Macro Precision: {self.best_val_results['macro_precision']:.4f}")
            logger.info(f"  Macro Recall: {self.best_val_results['macro_recall']:.4f}")
            
            logger.info("\n  📊 VALIDATION F1 SCORES BY METRIC:")
            for metric, results in self.best_val_results['per_metric_results'].items():
                logger.info(f"    {metric}: {results['f1']:.4f}")
            
            logger.info("\n  Per-Metric Detailed Results:")
            for metric, results in self.best_val_results['per_metric_results'].items():
                logger.info(f"    {metric}:")
                logger.info(f"      Accuracy: {results['accuracy']:.4f}")
                logger.info(f"      Precision: {results['precision']:.4f}")
                logger.info(f"      Recall: {results['recall']:.4f}")
                logger.info(f"      F1: {results['f1']:.4f}")
        
        logger.info("="*80)


class MetricsTrainer:
    """Main trainer class for fine-tuning Llama3.1-8b with LoRA."""
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_tokenizer(self):
        """Initialize tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Set padding side to 'left' for decoder-only models (like Llama)
        self.tokenizer.padding_side = 'left'
        logger.info("Set tokenizer padding_side to 'left' for decoder-only model")
            
        # Validate token IDs are within expected range
        vocab_size = len(self.tokenizer)
        if self.tokenizer.pad_token_id >= vocab_size:
            logger.error(f"pad_token_id ({self.tokenizer.pad_token_id}) >= vocab_size ({vocab_size})")
            raise ValueError("Invalid pad_token_id")
            
        if self.tokenizer.eos_token_id >= vocab_size:
            logger.error(f"eos_token_id ({self.tokenizer.eos_token_id}) >= vocab_size ({vocab_size})")
            raise ValueError("Invalid eos_token_id")
            
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        logger.info(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        logger.info(f"EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        logger.info(f"Padding side: {self.tokenizer.padding_side}")
        
        # Test tokenization to ensure it works
        test_text = "Test tokenization"
        test_tokens = self.tokenizer.encode(test_text)
        logger.info(f"Test tokenization successful: '{test_text}' -> {len(test_tokens)} tokens")
    
    def setup_model(self):
        """Initialize model with LoRA configuration."""
        logger.info("Loading base model...")
        
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16 if self.args.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_rank,  # LoRA rank
            lora_alpha=self.args.lora_alpha,  # LoRA alpha
            lora_dropout=self.args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],  # Target all linear layers in Llama
            bias="none",
        )
        
        # Apply LoRA to the model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing on the PEFT model
        if self.args.bf16:
            self.peft_model.enable_input_require_grads()
        
        self.peft_model.print_trainable_parameters()
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        logger.info(f"Model loaded with LoRA rank {self.args.lora_rank}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """Setup training and validation datasets."""
        logger.info("Preparing datasets...")
        
        dataset_handler = MetricsDataset(
            tokenizer=self.tokenizer,
            max_length=self.args.max_length
        )
        
        # Load training data
        self.train_dataset = dataset_handler.prepare_dataset(self.args.train_data_path)
        # Also keep the raw training data for evaluation
        self.train_dataset_raw = Dataset.from_dict({
            'instruction': [item['instruction'] for item in dataset_handler.load_data(self.args.train_data_path)],
            'input': [item['input'] for item in dataset_handler.load_data(self.args.train_data_path)],
            'output': [item['output'] for item in dataset_handler.load_data(self.args.train_data_path)]
        })
        logger.info(f"Training samples: {len(self.train_dataset)}")
        
        # Load validation data
        if self.args.val_data_path:
            self.val_dataset = dataset_handler.prepare_dataset(self.args.val_data_path)
            # Also keep the raw validation data for evaluation
            self.val_dataset_raw = Dataset.from_dict({
                'instruction': [item['instruction'] for item in dataset_handler.load_data(self.args.val_data_path)],
                'input': [item['input'] for item in dataset_handler.load_data(self.args.val_data_path)],
                'output': [item['output'] for item in dataset_handler.load_data(self.args.val_data_path)]
            })
            logger.info(f"Validation samples: {len(self.val_dataset)}")
        else:
            self.val_dataset = None
            self.val_dataset_raw = None
            
        # Define metric names
        self.metric_names = [
            "One-size-fits-all Ableism", "Infantilization", "Technoableism",
            "Anticipated Ableism", "Ability Saviorism", "Tokenism",
            "Inspiration Porn", "Superhumanization Harm"
        ]
    
    def setup_training_args(self):
        """Setup training arguments."""
        self.training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            max_grad_norm=self.args.max_grad_norm,
            lr_scheduler_type="cosine",
            warmup_ratio=self.args.warmup_ratio,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_steps=None,  # Disabled: validation evaluation handled by custom callback
            eval_strategy="no",  # Disabled: validation evaluation handled by custom callback
            save_strategy="steps",
            load_best_model_at_end=False,  # Disabled: best model selection handled by custom callback
            metric_for_best_model=None,  # Disabled: best model selection handled by custom callback
            greater_is_better=False,
            bf16=self.args.bf16,
            gradient_checkpointing=False,  # Disable gradient checkpointing to avoid issues with LoRA
            dataloader_drop_last=True,
            dataloader_num_workers=8,  # Reduce workers to save memory
            ddp_find_unused_parameters=False,  # Optimize DDP performance
            report_to="wandb" if self.args.use_wandb else "none",
            run_name=f"llama3_metrics_lora_r{self.args.lora_rank}_lr{self.args.learning_rate}",
            remove_unused_columns=False,
            seed=42,  # Set seed for reproducibility
            disable_tqdm=False,  # Enable progress bars
            dataloader_pin_memory=True,  # Optimize data loading
            prediction_loss_only=False,  # Show more detailed progress
            include_inputs_for_metrics=True,  # Enable detailed metrics
        )
    
    def setup_trainer(self):
        """Setup the trainer."""
        # Create a completely custom data collator to handle variable lengths
        class RobustDataCollator:
            def __init__(self, tokenizer, pad_to_multiple_of=8):
                self.tokenizer = tokenizer
                self.pad_to_multiple_of = pad_to_multiple_of
                
            def __call__(self, features):
                if not features:
                    return {}
                
                # Validate features
                required_keys = {"input_ids", "attention_mask", "labels"}
                for i, feature in enumerate(features):
                    if not all(key in feature for key in required_keys):
                        raise ValueError(f"Feature {i} missing required keys. Expected: {required_keys}, Got: {list(feature.keys())}")
                
                # Extract all the data
                input_ids = [f["input_ids"] for f in features]
                attention_masks = [f["attention_mask"] for f in features]
                labels = [f["labels"] for f in features]
                
                # Validate that all sequences have the same length for each feature
                for i, (ids, mask, label) in enumerate(zip(input_ids, attention_masks, labels)):
                    if len(ids) != len(mask) or len(ids) != len(label):
                        raise ValueError(f"Feature {i}: input_ids length ({len(ids)}) != attention_mask length ({len(mask)}) or labels length ({len(label)})")
                
                # Find the maximum length in this batch
                max_length = max(len(ids) for ids in input_ids)
                
                # Pad to multiple of pad_to_multiple_of for efficiency
                if self.pad_to_multiple_of > 0:
                    max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
                
                # Pad all sequences to the same length
                batch_input_ids = []
                batch_attention_masks = []
                batch_labels = []
                
                for i, (ids, mask, label) in enumerate(zip(input_ids, attention_masks, labels)):
                    # Calculate padding length
                    padding_length = max_length - len(ids)
                    
                    # Pad input_ids
                    padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                    batch_input_ids.append(padded_ids)
                    
                    # Pad attention_mask
                    padded_mask = mask + [0] * padding_length
                    batch_attention_masks.append(padded_mask)
                    
                    # Pad labels (use -100 for padding tokens to ignore in loss)
                    padded_labels = label + [-100] * padding_length
                    batch_labels.append(padded_labels)
                
                # Convert to tensors
                import torch
                batch = {
                    "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
                    "labels": torch.tensor(batch_labels, dtype=torch.long)
                }
                
                return batch
        
        data_collator = RobustDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8
        )
        
        # Add debugging information
        logger.info(f"Data collator configured with pad_token_id: {self.tokenizer.pad_token_id}")
        logger.info(f"Data collator will pad to multiple of: 8")
        
        # Test the data collator with a small sample to ensure it works
        try:
            # Create a small test batch
            test_features = [
                {
                    "input_ids": [1, 2, 3, 4, 5],
                    "attention_mask": [1, 1, 1, 1, 1],
                    "labels": [1, 2, 3, 4, 5]
                },
                {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                    "labels": [1, 2, 3]
                }
            ]
            test_batch = data_collator(test_features)
            logger.info(f"Data collator test successful. Batch shape: {test_batch['input_ids'].shape}")
        except Exception as e:
            logger.error(f"Data collator test failed: {e}")
            raise
        
        # Create custom callback for evaluation
        metrics_callback = MetricsCallback(
            trainer=None,  # Will be set after trainer creation
            val_dataset_raw=self.val_dataset_raw,
            train_dataset_raw=None,  # Disable training set evaluation
            metric_names=self.metric_names,
            eval_samples=self.args.eval_samples
        )
        
        self.trainer = Trainer(
            model=self.peft_model,
            args=self.training_args,
            train_dataset=self.train_dataset,  # Only training set used for training
            eval_dataset=None,  # Validation evaluation happens via custom callback, not Trainer's eval
            data_collator=data_collator,
            processing_class=self.tokenizer,
            callbacks=[metrics_callback],
        )
        
        # Set trainer reference in callback
        metrics_callback.trainer = self.trainer
        self.metrics_callback = metrics_callback
    
    def train(self):
        """Start training."""
        logger.info("Starting training...")
        logger.info(f"Training arguments: {self.training_args}")
        
        # Start training
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        # Print best results achieved during training
        self.metrics_callback.print_best_results()
        
        logger.info(f"Training completed! Model saved to {self.args.output_dir}")
        logger.info(f"Best adapter saved to: {os.path.join(self.args.output_dir, 'best_adapter')}")
    
    def save_adapter(self):
        """Save only the LoRA adapter."""
        adapter_path = os.path.join(self.args.output_dir, "adapter")
        self.peft_model.save_pretrained(adapter_path)
        logger.info(f"LoRA adapter saved to {adapter_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama3.1-8b with LoRA for metrics detection")
    
    # Model and data paths
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--train_data_path", type=str, required=True,
                       help="Path to training data JSON file")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="Path to validation data JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for the fine-tuned model")
    
    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank (recommended: 64 for comprehensive coverage)")
    parser.add_argument("--lora_alpha", type=float, default=128.0,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0,
                       help="LoRA dropout rate")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=float, default=3.0,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Regularization
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio for learning rate scheduler")
    
    # Training configuration
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Model saving frequency")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation frequency")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use BF16 mixed precision training")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="llama3-metrics-detection",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity/team name")
    parser.add_argument("--eval_samples", type=int, default=100,
                       help="Number of samples to use for evaluation during training")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        try:
            # Create organized run name and tags based on key parameters with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"lr{args.learning_rate}_rank{args.lora_rank}_bf16_ep{args.epochs}_{timestamp}"
        
            # Create tags for easy filtering
            tags = [
                "llama3", "lora", "metrics-detection", "fine-tuning",
                f"lr_{args.learning_rate}", f"rank_{args.lora_rank}", 
                "bf16", f"epochs_{args.epochs}"
            ]
            
            # Start a new wandb run to track this script
            wandb_init_kwargs = {
                # Set the wandb project where this run will be logged
                "project": args.wandb_project,
                # Track hyperparameters and run metadata
                "config": {
                    # Model configuration
                    "model_name": args.model_name,
                    "architecture": "Llama3.1-8B-Instruct",
                    "bf16": args.bf16,
                    "max_length": args.max_length,
                    
                    # LoRA configuration
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    
                    # Training configuration
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "weight_decay": args.weight_decay,
                    "max_grad_norm": args.max_grad_norm,
                    "warmup_ratio": args.warmup_ratio,
                    
                    # Data configuration
                    "dataset": "metrics_detection",
                    "eval_samples": args.eval_samples,
                    
                    # Output configuration
                    "output_dir": args.output_dir
                },
                # Set the run name for better organization
                "name": run_name,
                # Add tags for better filtering
                "tags": tags,
                # Add notes for additional context
                "notes": f"LoRA fine-tuning with rank={args.lora_rank}, lr={args.learning_rate}, bf16={args.bf16} (started at {timestamp})",
                # Add settings to prevent logging conflicts
                "settings": wandb.Settings(
                    console="off"  # Disable console output from wandb
                )
            }
            
            # Add entity only if specified and not None
            if args.wandb_entity and args.wandb_entity != "None":
                wandb_init_kwargs["entity"] = args.wandb_entity
                
            run = wandb.init(**wandb_init_kwargs)
            logger.info("Wandb initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            logger.warning("Continuing without wandb logging...")
            args.use_wandb = False
    
    # Check CUDA availability and GPU setup
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow on CPU.")
    else:
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available with {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"GPU {i}: {gpu_name}")
        
        if gpu_count > 1:
            logger.info("Multi-GPU training will be automatically enabled by Transformers Trainer")
    
    # Initialize trainer
    trainer = MetricsTrainer(args)
    
    # Setup components
    trainer.setup_tokenizer()
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_training_args()
    trainer.setup_trainer()
    
    # Start training
    trainer.train()
    trainer.save_adapter()
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
        logger.info("Wandb run completed and logged successfully!")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
