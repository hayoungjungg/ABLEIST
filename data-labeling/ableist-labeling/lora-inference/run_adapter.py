#!/usr/bin/env python3
"""
Evaluate uploaded LoRA adapters from Hugging Face Hub (e.g., hayoungjung/llama3.1-8b-adapter-ABLEist-detection)
on the validation set, test set, and more. Can be used for inference.

Usage:
    # For LoRA adapters (like hayoungjung/llama3.1-8b-adapter-ABLEist-detection)
    python evaluate_uploaded_adapter.py --model_name hayoungjung/llama3.1-8b-adapter-ABLEist-detection
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import yaml for metric parsing
import yaml
from typing import Dict, List, Any

def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def evaluate_model_on_dataset(model, tokenizer, data, dataset_name, output_dir):
    """
    Evaluate model on a dataset. This is a wrapper that uses the simplified evaluation.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        data: List of data samples
        dataset_name: Name of the dataset
        output_dir: Output directory (not used in simplified version)
    
    Returns:
        Dictionary with evaluation results
    """
    metric_names = [
        "One-size-fits-all Ableism",
        "Infantilization", 
        "Technoableism",
        "Anticipated Ableism",
        "Ability Saviorism",
        "Tokenism",
        "Inspiration Porn",
        "Superhumanization Harm"
    ]
    
    return evaluate_model_simplified(model, tokenizer, data, metric_names, dataset_name)

def compute_comprehensive_metrics(predictions: List[Dict], metric_names: List[str], 
                                   dataset_name: str, metric_exclusions: Dict[str, List[int]] = None):
    """
    Calculate comprehensive metrics including macro/weighted F1, accuracy, precision, recall.
    
    Args:
        predictions: List of prediction dictionaries with 'predicted_metrics', 'true_metrics', 'original_index'
        metric_names: List of metric names
        dataset_name: Name of the dataset
        metric_exclusions: Dictionary mapping metric names to lists of indices to exclude (e.g., few-shot examples
                           in the LLM evaluations to ensure comparisons across exact same examples)
    
    Returns:
        Dictionary with comprehensive metrics
    """
    # Convert predictions to arrays
    all_pred_values = []
    all_true_values = []
    
    for pred in predictions:
        pred_metrics = pred['predicted_metrics']
        true_metrics = pred['true_metrics']
        original_index = pred.get('original_index', 0)
        
        # Handle exclusions
        pred_values = []
        true_values = []
        
        for metric in metric_names:
            pred_val = pred_metrics.get(metric, 0)
            true_val = true_metrics.get(metric, 0)
            
            # Check if this metric should be excluded for this sample
            should_exclude = False
            if metric_exclusions and metric in metric_exclusions:
                if original_index in metric_exclusions[metric]:
                    should_exclude = True
            
            if not should_exclude:
                pred_values.append(pred_val)
                true_values.append(true_val)
            else:
                # Use -1 to mark excluded (will be filtered out)
                pred_values.append(-1)
                true_values.append(-1)
        
        all_pred_values.append(pred_values)
        all_true_values.append(true_values)
    
    # Convert to numpy arrays
    predictions_array = np.array(all_pred_values)
    ground_truth_array = np.array(all_true_values)
    
    # Filter out excluded samples (-1 values)
    valid_mask = (predictions_array != -1) & (ground_truth_array != -1)
    
    # Overall metrics (flattened across all valid samples and metrics)
    valid_predictions_flat = predictions_array[valid_mask]
    valid_ground_truth_flat = ground_truth_array[valid_mask]
    
    if len(valid_predictions_flat) > 0:
        overall_accuracy = accuracy_score(valid_ground_truth_flat, valid_predictions_flat)
        overall_precision, overall_recall, overall_f1_macro, _ = precision_recall_fscore_support(
            valid_ground_truth_flat, valid_predictions_flat, average='macro', zero_division=0
        )
        _, _, overall_f1_weighted, _ = precision_recall_fscore_support(
            valid_ground_truth_flat, valid_predictions_flat, average='weighted', zero_division=0
        )
    else:
        overall_accuracy = overall_precision = overall_recall = overall_f1_macro = overall_f1_weighted = 0.0
    
    # Per-metric metrics
    per_metric_results = {}
    for i, metric in enumerate(metric_names):
        metric_pred = predictions_array[:, i]
        metric_true = ground_truth_array[:, i]
        
        # Filter out excluded samples for this metric
        metric_valid_mask = (metric_pred != -1) & (metric_true != -1)
        metric_pred_valid = metric_pred[metric_valid_mask]
        metric_true_valid = metric_true[metric_valid_mask]
        
        if len(metric_pred_valid) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                metric_true_valid, metric_pred_valid, average='binary', zero_division=0
            )
            accuracy = accuracy_score(metric_true_valid, metric_pred_valid)
            
            per_metric_results[metric] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'valid_samples': len(metric_pred_valid),
                'total_samples': len(metric_pred),
                'excluded_samples': len(metric_pred) - len(metric_pred_valid)
            }
        else:
            per_metric_results[metric] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'valid_samples': 0,
                'total_samples': len(metric_pred),
                'excluded_samples': len(metric_pred)
            }
    
    return {
        'dataset_name': dataset_name,
        'total_samples': len(predictions),
        'valid_samples': len(valid_predictions_flat),
        'overall_accuracy': overall_accuracy,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_macro': overall_f1_macro,
        'overall_f1_weighted': overall_f1_weighted,
        'per_metric_results': per_metric_results,
        'metric_exclusions': metric_exclusions or {}
    }

def load_uploaded_model(model_name, device_map="auto"):
    """
    Load a LoRA adapter from Hugging Face Hub.
    
    Args:
        model_name: Hugging Face adapter name (e.g., 'hayoungjung/llama3.1-8b-adapter-ABLEist-detection')
        device_map: Device mapping for model loading
    
    Returns:
        model, tokenizer, model_info
    """
    logger.info(f"Loading LoRA adapter: {model_name}")
    
    try:
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_name)
        
        # Get adapter config
        try:
            peft_config = PeftConfig.from_pretrained(model_name)
            model_info = {
                "type": "LoRA Adapter",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "adapter_name": model_name,
                "peft_config": peft_config
            }
        except:
            model_info = {
                "type": "LoRA Adapter",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "adapter_name": model_name,
                "peft_config": None
            }
        
        logger.info("✅ Model loaded successfully!")
        return model, tokenizer, model_info
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def run_evaluation_with_existing_framework(model, tokenizer, data_path, dataset_name, output_dir):
    """Run evaluation using the comprehensive framework."""
    logger.info(f"Running evaluation on {dataset_name}...")
    
    try:
        # Load data
        data = load_data(data_path)
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        
        # Run evaluation
        results = evaluate_model_on_dataset(
            model, tokenizer, data, dataset_name, output_dir
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}")
        raise

def run_simplified_evaluation(model, tokenizer, data_path, dataset_name, output_dir):
    """Run simplified evaluation when existing framework is not available."""
    logger.info(f"Running simplified evaluation on {dataset_name}...")
    
    try:
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        
        # Define metric names
        metric_names = [
            "One-size-fits-all Ableism",
            "Infantilization", 
            "Technoableism",
            "Anticipated Ableism",
            "Ability Saviorism",
            "Tokenism",
            "Inspiration Porn",
            "Superhumanization Harm"
        ]
        
        # Run evaluation
        results = evaluate_model_simplified(model, tokenizer, data, metric_names, dataset_name)
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to run simplified evaluation: {e}")
        raise

def evaluate_model_simplified(model, tokenizer, data, metric_names, dataset_name, max_samples=None):
    """Simplified evaluation function."""
    
    if max_samples:
        data = data[:max_samples]
    
    logger.info(f"Evaluating {len(data)} samples...")
    
    predictions = []
    ground_truth = []
    
    for i, sample in enumerate(tqdm(data, desc=f"Evaluating {dataset_name}")):
        try:
            # Create prompt
            prompt = create_prompt(sample)
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            # Check if response is empty or too short
            if not response or len(response.strip()) < 10:
                logger.warning(f"Sample {i}: Empty or very short response from model: '{response[:100]}...'")
            
            # Extract metrics
            pred_metrics = extract_metrics_from_response(response, metric_names)
            # Ground truth is in the 'output' field as YAML string, need to parse it
            true_metrics = extract_metrics_from_response(sample.get('output', ''), metric_names)
            
            # Log if any metrics are missing from predictions
            missing_pred_metrics = [metric for metric, value in pred_metrics.items() if value == 0 and metric not in response]
            if missing_pred_metrics:
                logger.warning(f"Sample {i}: Missing metrics in prediction: {missing_pred_metrics}")
            
            # Log if any metrics are missing from ground truth
            missing_true_metrics = [metric for metric, value in true_metrics.items() if value == 0 and metric not in sample.get('output', '')]
            if missing_true_metrics:
                logger.warning(f"Sample {i}: Missing metrics in ground truth: {missing_true_metrics}")
            
            # Store with original index for comprehensive metrics
            predictions.append({
                'predicted_metrics': pred_metrics,
                'true_metrics': true_metrics,
                'original_index': sample.get('index', i)
            })
            ground_truth.append(true_metrics)
            
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            # Add default predictions
            pred_metrics = {metric: 0 for metric in metric_names}
            true_metrics = extract_metrics_from_response(sample.get('output', ''), metric_names)
            logger.warning(f"Sample {i}: Using default predictions (all 0s) due to processing error")
            predictions.append({
                'predicted_metrics': pred_metrics,
                'true_metrics': true_metrics,
                'original_index': sample.get('index', i)
            })
            ground_truth.append(true_metrics)
    
    # Compute metrics - extract just the metrics dicts for simplified computation
    pred_metrics_list = [p['predicted_metrics'] for p in predictions]
    results = compute_metrics_simplified(pred_metrics_list, ground_truth, metric_names, dataset_name)
    
    # Add the full predictions list for comprehensive metrics
    results['all_predictions'] = predictions
    
    # Add summary of missing metrics
    total_samples = len(predictions)
    samples_with_missing_preds = sum(1 for pred in pred_metrics_list if any(v == 0 for v in pred.values()))
    samples_with_missing_truth = sum(1 for truth in ground_truth if any(v == 0 for v in truth.values()))
    
    logger.info(f"Evaluation summary for {dataset_name}:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Samples with missing predictions: {samples_with_missing_preds}")
    logger.info(f"  Samples with missing ground truth: {samples_with_missing_truth}")
    
    return results

def extract_metrics_from_response(response_text, metric_names):
    """Extract metric values from model response (YAML format)."""
    import yaml
    
    metrics = {}
    
    try:
        # First try to find YAML in code blocks
        yaml_start = response_text.find('```yaml')
        if yaml_start != -1:
            yaml_start += 7  # Skip '```yaml'
            yaml_end = response_text.find('```', yaml_start)
            if yaml_end != -1:
                yaml_str = response_text[yaml_start:yaml_end].strip()
                parsed_metrics = yaml.safe_load(yaml_str)
                if isinstance(parsed_metrics, dict):
                    for metric in metric_names:
                        metrics[metric] = parsed_metrics.get(metric, 0)
                    return metrics
        
        # Try to extract YAML-like content from the beginning of the response
        lines = response_text.strip().split('\n')
        yaml_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and code block markers
            if not line or line.startswith('```'):
                continue
            # Look for metric: value pattern
            if ':' in line and any(metric in line for metric in metric_names):
                yaml_lines.append(line)
        
        if yaml_lines:
            yaml_str = '\n'.join(yaml_lines)
            try:
                parsed_metrics = yaml.safe_load(yaml_str)
                if isinstance(parsed_metrics, dict):
                    for metric in metric_names:
                        metrics[metric] = parsed_metrics.get(metric, 0)
                    return metrics
            except yaml.YAMLError:
                # YAML parsing failed, continue to fallback
                print(f"YAML parsing failed for: {yaml_str}")
                pass
        
        # Fallback: try to extract individual metrics
        for metric in metric_names:
            # Look for metric in response
            patterns = [
                f"{metric}:",
                f"'{metric}':",
                f'"{metric}":',
            ]
            
            found = False
            for pattern in patterns:
                if pattern in response_text:
                    # Extract value after pattern
                    start_idx = response_text.find(pattern) + len(pattern)
                    remaining = response_text[start_idx:].strip()
                    
                    # Look for 0 or 1
                    if remaining.startswith('0') or remaining.startswith('1'):
                        value = int(remaining[0])
                        metrics[metric] = value
                        found = True
                        break
            
        if not found:
            # Default to 0 if not found
            logger.warning(f"Metric '{metric}' not found in response, defaulting to 0")
            metrics[metric] = 0
        
    except Exception as e:
        logger.warning(f"Error parsing metrics from response: {e}")
        # Fallback to default values
        for metric in metric_names:
            logger.warning(f"Metric '{metric}' not found in response due to parsing error, defaulting to 0")
            metrics[metric] = 0
    
    return metrics

def create_prompt(sample):
    """Create prompt for evaluation using the same format as training."""
    # Use the same format as the training/evaluation scripts
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    return prompt

def compute_metrics_simplified(predictions, ground_truth, metric_names, dataset_name):
    """Compute evaluation metrics."""
    
    # Convert to arrays for sklearn
    y_true = []
    y_pred = []
    
    for pred, true in zip(predictions, ground_truth):
        for metric in metric_names:
            y_true.append(true[metric])
            y_pred.append(pred[metric])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Compute per-metric metrics
    metric_results = {}
    for i, metric in enumerate(metric_names):
        start_idx = i * len(predictions)
        end_idx = (i + 1) * len(predictions)
        
        metric_true = y_true[start_idx:end_idx]
        metric_pred = y_pred[start_idx:end_idx]
        
        metric_accuracy = accuracy_score(metric_true, metric_pred)
        metric_precision, metric_recall, metric_f1, _ = precision_recall_fscore_support(
            metric_true, metric_pred, average='macro', zero_division=0
        )
        
        metric_results[metric] = {
            'accuracy': metric_accuracy,
            'precision': metric_precision,
            'recall': metric_recall,
            'f1': metric_f1,
            'samples': len(metric_true)
        }
    
    results = {
        'dataset_name': dataset_name,
        'total_samples': len(predictions),
        'valid_samples': len(predictions),
        'overall_accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'metric_results': metric_results,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return results

def print_summary(all_results):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("🎯 UPLOADED ADAPTER EVALUATION SUMMARY")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\n📊 {dataset_name.upper()} RESULTS:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Valid samples: {results['valid_samples']}")
        print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"  Macro Precision: {results['macro_precision']:.4f}")
        print(f"  Macro Recall: {results['macro_recall']:.4f}")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
    
    print("="*80)

def save_results(all_results, output_dir, model_name, run_name):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"uploaded_adapter_evaluation_{timestamp}_results.json")
    
    # Prepare results for saving
    save_data = {
        'model_name': model_name,
        'run_name': run_name,
        'evaluation_time': datetime.now().isoformat(),
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    return results_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate uploaded LoRA adapter from Hugging Face Hub")
    parser.add_argument('--model_name', type=str, required=True,
                       help='Hugging Face adapter name (e.g., hayoungjung/llama3.1-8b-adapter-ABLEist-detection)')
    parser.add_argument('--val_data_path', type=str,
                       default='/n/fs/peas-lab/hayoung/chast-abl/script_prep/data/my_val.json',
                       help='Path to validation data')
    parser.add_argument('--test_data_path', type=str,
                       default='/n/fs/peas-lab/hayoung/chast-abl/script_prep/data/my_test.json',
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for this evaluation')
    parser.add_argument('--device_map', type=str, default='auto',
                       help='Device mapping for model loading')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"uploaded_adapter_eval_{timestamp}"
    
    logger.info("🧪 Comprehensive Evaluation of Uploaded LoRA Adapter")
    logger.info("="*50)
    logger.info(f"Adapter: {args.model_name}")
    logger.info(f"Model Type: LoRA Adapter")
    logger.info(f"Validation Data: {args.val_data_path}")
    logger.info(f"Test Data: {args.test_data_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Run Name: {args.run_name}")
    logger.info("="*50)
    
    try:
        # Load model with the finetuned LoRA adapter 
        model, tokenizer, model_info = load_uploaded_model(args.model_name, args.device_map)
        
        all_results = {}
        
        # Evaluate on validation set
        if os.path.exists(args.val_data_path):
            logger.info("Evaluating on validation set...")
            val_results = run_evaluation_with_existing_framework(
                model, tokenizer, args.val_data_path, 'validation', args.output_dir
            )
            all_results['validation'] = val_results
        else:
            logger.warning(f"Validation data not found: {args.val_data_path}")
        
        # Evaluate on test set
        if os.path.exists(args.test_data_path):
            logger.info("Evaluating on test set...")
            test_results = run_evaluation_with_existing_framework(
                model, tokenizer, args.test_data_path, 'main_test', args.output_dir
            )
            all_results['main_test'] = test_results
        else:
            logger.warning(f"Test data not found: {args.test_data_path}")
        
        # Print summary
        print_summary(all_results)
        
        # Save results
        results_file = save_results(all_results, args.output_dir, args.model_name, args.run_name)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {results_file}")
        
        # You can now use extract_f1_scores.py on the results file
        print(f"\n💡 To analyze these results with the extract_f1_scores.py script:")
        print(f"python extract_f1_scores.py --results_file {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
