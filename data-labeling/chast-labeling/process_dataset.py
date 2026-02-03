#!/usr/bin/env python3
"""
Process the labeled_ableism_complete_dataset.csv file to generate prompts and run CHAST inference.
"""

import pandas as pd
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from chast_inference import CHASTInference
from script_prep.prompt import get_chast_prompts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df

def create_prompts_from_dataset(df: pd.DataFrame) -> list:
    """Create prompts from the dataset using get_chast_prompts function."""
    prompts = []
    prompt_data = []
    
    logger.info("Creating prompts from dataset...")
    
    for idx, row in df.iterrows():
        # Get the response (conversation) from the dataset
        response = row['Response']
        
        # Create prompt using the get_chast_prompts function
        prompt = get_chast_prompts(response)
        
        # Store both the prompt and metadata
        prompt_data.append({
            'index': row['index'],
            'original_response': response,
            'prompt': prompt
        })
        prompts.append(prompt)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1} prompts...")
    
    logger.info(f"Created {len(prompts)} prompts")
    logger.info(prompts[0])
    logger.info(prompts[1000])
    return prompts, prompt_data

def run_inference(model, prompts: list, prompt_data: list, output_file: str, max_new_tokens: int = 512) -> list:
    """Run inference on the prompts using the CHAST model with periodic saving."""
    logger.info(f"Running inference on {len(prompts)} prompts...")
        
    outputs = []
    response = ""
    save_interval = 50  # Save every 50 prompts
    
    # Create intermediate save file (separate from final output)
    intermediate_file = output_file.replace('.json', '_intermediate.json')
    
    for i, prompt in enumerate(prompts):
        if i % 5 == 1:
            print(i)
            print(response)

        logger.info(f"Processing prompt {i+1}/{len(prompts)}")
        
        try:
            # Use greedy decoding (temperature=0, do_sample=False)
            response = model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0,
                do_sample=False
            )
            outputs.append(response)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1} prompts")
            
            # Save periodically to intermediate file
            if (i + 1) % save_interval == 0:
                logger.info(f"Saving intermediate results at prompt {i + 1}...")
                save_results(prompt_data[:i+1], outputs, intermediate_file)
                
        except Exception as e:
            logger.error(f"Error processing prompt {i+1}: {e}")
            outputs.append(f"ERROR: {str(e)}")
    
    logger.info("Inference completed")
    return outputs

def save_results(prompt_data: list, outputs: list, output_file: str):
    """Save the results to a JSON file."""
    logger.info(f"Saving results to {output_file}")
    
    results = []
    for i, (data, output) in enumerate(zip(prompt_data, outputs)):
        result = {
            'index': data['index'],
            'original_response': data['original_response'],
            'prompt': data['prompt'],
            'chast_output': output
        }
        results.append(result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_file}")


def main():
    # File paths with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = "/n/fs/peas-lab/hayoung/chast-abl/script_prep/labeled_ableism_complete_dataset.csv"
    output_path = f"/n/fs/peas-lab/hayoung/chast-abl/chast_inference_results_{timestamp}.json"
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Create prompts
    prompts, prompt_data = create_prompts_from_dataset(df)
    
    # Initialize CHAST model
    logger.info("Initializing CHAST model...")
    model = CHASTInference(
        base_model="lmsys/vicuna-13b-v1.5-16k",
        adapter_repo="SocialCompUW/CHAST",
        use_quantization=False
    )
    
    # Load model
    logger.info("Loading model...")
    model.load_model()
    
    # Test model first
    logger.info("Testing model...")
    test_response = model.test_model()
    logger.info(f"Test response: {test_response}")
    
    # Run inference
    outputs = run_inference(model, prompts, prompt_data, output_path, max_new_tokens=1024)
    
    # Save final results
    save_results(prompt_data, outputs, output_path)
    
    # Clean up intermediate file
    intermediate_file = output_path.replace('.json', '_intermediate.json')
    if Path(intermediate_file).exists():
        Path(intermediate_file).unlink()
        logger.info(f"Cleaned up intermediate file: {intermediate_file}")
    
    logger.info("Processing completed!")

if __name__ == "__main__":
    main()
