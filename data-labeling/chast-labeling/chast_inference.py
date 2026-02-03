#!/usr/bin/env python3
"""
CHAST Inference Script
A modular script for running inference with the CHAST model.
"""

import os
import argparse
import json
import logging
import sys
import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CHASTInference:
    """CHAST model inference class with modular functionality."""
    
    def __init__(self, 
                 base_model: str = "lmsys/vicuna-13b-v1.5-16k",
                 adapter_repo: str = "SocialCompUW/CHAST",
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 use_quantization: bool = True,
                 device_map: str = "auto"):
        """
        Initialize CHAST inference model.
        
        Args:
            base_model: HuggingFace model identifier
            adapter_repo: PEFT adapter repository
            device: Device to use (auto-detected if None)
            dtype: Data type for model (auto-selected if None)
            use_quantization: Whether to use 4-bit quantization
            device_map: Device mapping strategy (auto/balanced/sequential)
        """
        self.base_model = base_model
        self.adapter_repo = adapter_repo
        self.device = device
        self.dtype = dtype
        self.use_quantization = use_quantization
        self.device_map = device_map
        
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        # Setup device and dtype
        self._setup_device_and_dtype()
        
    def _setup_device_and_dtype(self):
        """Setup device and data type based on available hardware."""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_count = torch.cuda.device_count()
                logger.info(f"CUDA is available. Number of GPUs: {gpu_count}")
                for i in range(gpu_count):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            else:
                self.device = "cpu"
                logger.warning("CUDA not available. Using CPU.")
        
        if self.dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                logger.info("Using bfloat16 precision")
            else:
                self.dtype = torch.float16
                logger.info("Using float16 precision")
    
    def load_model(self, force_reload: bool = False):
        """
        Load the base model and adapter.
        
        Args:
            force_reload: Force reload even if already loaded
        """
        if self.is_loaded and not force_reload:
            logger.info("Model already loaded. Use force_reload=True to reload.")
            return
        
        logger.info(f"Loading base model: {self.base_model}")
        
        # Setup quantization config
        bnb_config = None
        if self.use_quantization and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info("Loading base model...")
        model_kwargs = {
            "dtype": self.dtype,
        }
        
        # Setup device mapping for multi-GPU
        if self.device == "cuda":
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # Use specified device mapping strategy
                if self.device_map == "auto":
                    model_kwargs["device_map"] = "balanced"  # Default to balanced for multi-GPU
                elif self.device_map == "single":
                    model_kwargs["device_map"] = {"": 0}  # Use only GPU 0
                    logger.info("Using single GPU (GPU 0) for faster inference")
                else:
                    model_kwargs["device_map"] = self.device_map
                logger.info(f"Using {self.device_map} device mapping across {gpu_count} GPUs")
            else:
                model_kwargs["device_map"] = "auto"
                logger.info("Using auto device mapping for single GPU")
        else:
            model_kwargs["device_map"] = None
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs
        )
        base_model.eval()
        
        # Load adapter
        logger.info(f"Loading adapter: {self.adapter_repo}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_repo)
        self.model.eval()
        
        self.is_loaded = True
        logger.info("Model loaded successfully!")
        
        # Print model architecture
        #self._print_model_architecture()
        
        # Log GPU memory usage
        if self.device == "cuda":
            self._log_gpu_memory_usage()
    
    def _print_model_architecture(self):
        """Print model architecture and verify adapter integration."""
        if self.model is None:
            return
        
        logger.info("=" * 60)
        logger.info("MODEL ARCHITECTURE ANALYSIS")
        logger.info("=" * 60)
        
        # Print model type
        logger.info(f"Model Type: {type(self.model).__name__}")
        logger.info(f"Base Model: {self.base_model}")
        logger.info(f"Adapter: {self.adapter_repo}")
        
        # Check if it's a PEFT model
        if hasattr(self.model, 'base_model'):
            logger.info("✓ PEFT Model detected - Adapter is integrated")
            logger.info(f"Base Model Type: {type(self.model.base_model).__name__}")
            
            # Print adapter information
            if hasattr(self.model, 'peft_config'):
                logger.info("✓ PEFT Configuration found:")
                for key, config in self.model.peft_config.items():
                    logger.info(f"  - {key}: {config}")
            
            # Print model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total Parameters: {total_params:,}")
            logger.info(f"Trainable Parameters: {trainable_params:,}")
            logger.info(f"Trainable %: {(trainable_params/total_params)*100:.2f}%")
            
            # Print model structure
            logger.info("\nModel Structure:")
            logger.info("-" * 40)
            for name, module in self.model.named_children():
                logger.info(f"{name}: {type(module).__name__}")
                if hasattr(module, 'num_parameters'):
                    logger.info(f"  Parameters: {module.num_parameters():,}")
            
            # Check for adapter layers
            logger.info("\nAdapter Integration Check:")
            logger.info("-" * 40)
            adapter_found = False
            for name, module in self.model.named_modules():
                if 'lora' in name.lower() or 'adapter' in name.lower() or 'peft' in name.lower():
                    logger.info(f"✓ Adapter layer found: {name} ({type(module).__name__})")
                    adapter_found = True
            
            if not adapter_found:
                logger.warning("⚠ No obvious adapter layers found in model structure")
            else:
                logger.info("✓ Adapter layers successfully integrated")
                
        else:
            logger.warning("⚠ Not a PEFT model - adapter may not be properly loaded")
        
        # Print device information
        logger.info(f"\nDevice Information:")
        logger.info("-" * 40)
        if hasattr(self.model, 'device'):
            logger.info(f"Model Device: {self.model.device}")
        
        # Print quantization info
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'quantization_config'):
            logger.info(f"Quantization: {self.model.config.quantization_config}")
        else:
            logger.info("Quantization: Not detected")
            
        logger.info("=" * 60)
    
    def _log_gpu_memory_usage(self):
        """Log current GPU memory usage across all GPUs."""
        if not torch.cuda.is_available():
            return
        
        gpu_count = torch.cuda.device_count()
        logger.info("GPU Memory Usage:")
        for i in range(gpu_count):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    
    def generate(self, 
                 prompt: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0,
                 do_sample: bool = True,
                 **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (default: 0 for greedy)
            do_sample: Whether to use sampling (default: False for greedy)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generation config with optimized parameters
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Only add temperature if sampling is enabled
        if do_sample and temperature > 0:
            generation_config.temperature = temperature
        
        # Generate with error handling
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
        except RuntimeError as e:
            if "probability tensor" in str(e) or "device-side assert" in str(e):
                logger.warning("Generation failed with probability tensor error. Trying with greedy decoding...")
                # Fallback to greedy decoding (no sampling)
                fallback_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # Use greedy decoding
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=fallback_config
                    )
            else:
                raise e
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clear GPU cache after generation to free memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return generated_text.strip()
    
    def batch_generate(self, 
                      prompts: List[str],
                      max_new_tokens: int = 1024,
                      temperature: float = 0,
                      do_sample: bool = True,
                      **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input text prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (default: 0 for greedy)
            do_sample: Whether to use sampling (default: False for greedy)
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                **kwargs
            )
            results.append(result)
            
            # Log memory usage every 10 prompts
            if (i + 1) % 10 == 0 and self.device == "cuda":
                self._log_gpu_memory_usage()
        
        return results
    
    def clear_gpu_memory(self):
        """Manually clear GPU memory cache."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def inspect_model(self):
        """Manually inspect the model architecture and adapter integration."""
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return
        
        self._print_model_architecture()
        
        # Additional detailed inspection
        logger.info("\nDetailed Model Inspection:")
        logger.info("-" * 40)
        
        # Check for specific adapter types
        if hasattr(self.model, 'peft_config'):
            logger.info("PEFT Configuration Details:")
            for key, config in self.model.peft_config.items():
                logger.info(f"  {key}:")
                if hasattr(config, 'task_type'):
                    logger.info(f"    Task Type: {config.task_type}")
                if hasattr(config, 'inference_mode'):
                    logger.info(f"    Inference Mode: {config.inference_mode}")
                if hasattr(config, 'base_model_name_or_path'):
                    logger.info(f"    Base Model: {config.base_model_name_or_path}")
        
        # Check model state
        logger.info(f"\nModel State:")
        logger.info(f"  Training Mode: {self.model.training}")
        logger.info(f"  Eval Mode: {not self.model.training}")
        
        # Check for LoRA adapters specifically
        lora_layers = []
        for name, module in self.model.named_modules():
            if 'lora' in name.lower():
                lora_layers.append(name)
        
        if lora_layers:
            logger.info(f"\nLoRA Adapter Layers Found ({len(lora_layers)}):")
            for layer in lora_layers[:10]:  # Show first 10
                logger.info(f"  - {layer}")
            if len(lora_layers) > 10:
                logger.info(f"  ... and {len(lora_layers) - 10} more")
        else:
            logger.warning("No LoRA layers found - adapter may not be properly integrated")
    
    def reset_model_state(self):
        """Reset the model to eval mode and clear any problematic state."""
        if self.model is not None:
            self.model.eval()
            # Clear any cached states
            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None
            logger.info("Model state reset")
    
    def test_model(self, test_prompt: str = "Hello. Test Test. Respond with 'HELLO WORLD'") -> str:
        """Test the model with a simple prompt to verify it's working."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Testing model with prompt: '{test_prompt}'")
        # Reset model state before testing
        self.reset_model_state()
        try:
            response = self.generate(
                prompt=test_prompt,
                max_new_tokens=50,
                temperature=0,
                do_sample=True
            )
            logger.info("Model test successful!")
            return response
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            # Try one more time with the most basic generation possible
            logger.info("Trying with minimal generation parameters...")
            try:
                response = self.generate(
                    prompt=test_prompt,
                    max_new_tokens=20,
                    temperature=0,
                    do_sample=True
                )
                logger.info("Model test successful with minimal parameters!")
                return response
            except Exception as e2:
                logger.error(f"Model test failed even with minimal parameters: {e2}")
                raise e
    
    def save_outputs(self, outputs: List[str], output_file: str):
        """Save generated outputs to a file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, output in enumerate(outputs):
                f.write(f"Output {i+1}:\n{output}\n\n")
        logger.info(f"Outputs saved to {output_file}")


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file (one prompt per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_prompts_from_csv(file_path: str, column: str = "prompt") -> List[str]:
    """Load prompts from a CSV file."""
    df = pd.read_csv(file_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV. Available columns: {df.columns.tolist()}")
    return df[column].tolist()


def main():
    parser = argparse.ArgumentParser(description="CHAST Model Inference Script")
    
    # Model configuration
    parser.add_argument("--base-model", default="lmsys/vicuna-13b-v1.5-16k",
                       help="Base model identifier")
    parser.add_argument("--adapter-repo", default="SocialCompUW/CHAST",
                       help="PEFT adapter repository")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use")
    parser.add_argument("--device-map", choices=["auto", "balanced", "sequential", "single"], default="auto",
                       help="Device mapping strategy for multi-GPU (auto/balanced/sequential/single)")
    parser.add_argument("--no-quantization", action="store_true",
                       help="Disable 4-bit quantization")
    
    # Input/Output
    parser.add_argument("--prompt", type=str, help="Single prompt for inference")
    parser.add_argument("--prompt-file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--csv-file", type=str, help="CSV file containing prompts")
    parser.add_argument("--csv-column", default="prompt", help="Column name in CSV file")
    parser.add_argument("--output-file", type=str, help="Output file for results")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0,
                       help="Sampling temperature (default: 0 for greedy)")
    parser.add_argument("--do-sample", action="store_true",
                       help="Enable sampling (default: False for greedy decoding)")
    
    # Utility
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with a simple prompt")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test the model with a simple prompt and exit")
    parser.add_argument("--inspect-model", action="store_true",
                       help="Inspect model architecture and adapter integration")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine device
    device = None if args.device == "auto" else args.device
    
    # Initialize model
    logger.info("Initializing CHAST model...")
    model = CHASTInference(
        base_model=args.base_model,
        adapter_repo=args.adapter_repo,
        device=device,
        use_quantization=not args.no_quantization,
        device_map=args.device_map
    )
    
    # Load model
    logger.info("Loading model...")
    model.load_model()
    
    # Test only mode
    if args.test_only:
        logger.info("Running model test...")
        test_response = model.test_model()
        print(f"Test response: {test_response}")
        return
    
    # Inspect model mode
    if args.inspect_model:
        logger.info("Inspecting model architecture...")
        model.inspect_model()
        return
    
    # Determine prompts
    prompts = []
    
    if args.test_mode:
        prompts = ["Hello, how are you today?"]
        logger.info("Running in test mode with sample prompt")
    elif args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    elif args.csv_file:
        prompts = load_prompts_from_csv(args.csv_file, args.csv_column)
        logger.info(f"Loaded {len(prompts)} prompts from {args.csv_file}")
    else:
        logger.error("No input provided. Use --prompt, --prompt-file, --csv-file, or --test-mode")
        sys.exit(1)
    
    # Generate outputs
    logger.info(f"Generating responses for {len(prompts)} prompts...")
    
    # Generate outputs
    outputs = model.batch_generate(
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True
    )
    
    # Save or display results
    if args.output_file:
        model.save_outputs(outputs, args.output_file)
        logger.info(f"Results saved to {args.output_file}")
    else:
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n--- Prompt {i+1} ---")
            print(f"Input: {prompt}")
            print(f"Output: {output}")
            print("-" * 50)


if __name__ == "__main__":
    main()