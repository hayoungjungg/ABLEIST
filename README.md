# ABLEIST

This repo contains the code and data accompanying the preprint [ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios](https://arxiv.org/abs/2510.10998). This project outlines generating synthetic hiring conversations, labeling them for various forms of ableism, and analyzing the data, including evaluating baseline models.

## ABLEIST Overview

ABLEIST evaluates conversations between hiring managers to identify subtle forms of ableism, including:
- One-size-fits-all Ableism
- Infantilization
- Technoableism
- Anticipated Ableism
- Ability Saviorism
- Tokenism
- Inspiration Porn
- Superhumanization Harm

The project supports multiple labeling approaches: LLM-based prompting (zero-shot and few-shot) and fine-tuned inference using a [LoRA adapter](https://huggingface.co/hayoungjung/llama3.1-8b-adapter-ABLEist-detection). 

## Repository Structure

```
ABLEIST/
├── data/                          # Datasets and baseline results
│   ├── baselines/                 # Baseline evaluation results 
│   └── labeled_ableism_complete_dataset_filtered.csv     # ABLEIST-labeled data (see "Data Access" section below)
├── data-generation/               # Synthetic conversation generation
│   ├── generate_data.py           # Main data generation script
│   ├── llm_interface.py           # LLM API interface
│   └── prompt_template.txt        # Generation prompts
├── data-labeling/                 # Labeling approaches
│   ├── ableist-labeling/
│   │   ├── llms-prompting/        # LLM-based labeling (zero-shot/few-shot prompting)
│   │   └── lora-inference/        # Fine-tuning & inference using LLaMA models
│   └── chast-labeling/            # CHAST model inference (baseline)
└── data-analysis/                 # Analysis notebooks
    └── analysis.ipynb
```

## Quick Start

### Prerequisites

Set up environment variables (create a `.env` file), set an OpenAI API key, and securely use the API key downstream for the data generation or labeling. 

### Labeling ABLEIST 

- To use LLMs like `GPT-5-chat-latest` for labeling, use the evaluation notebook. See: `data-labeling/ableist-labeling/llms-prompting/evaluation.ipynb`. 
- To see the prompts and other helper functions for using LLMs to label, refer to `data-labeling/ableist-labeling/llms-prompting/utils`.
- To use fine-tuned Llama-3.1-8B model on [HuggingFace](https://huggingface.co/hayoungjung/llama3.1-8b-adapter-ABLEist-detection), use the inference script available at: `data-labeling/ableist-labeling/lora-inference/run_adapter.py`.
- To see our training script for the Llama-3.1-8B model, please see `data-labeling/ableist-labeling/lora-inference/train_llama3_metrics.py`.

## Data Access

As discussed in *Ethical Considerations* of the paper, we are not publicly releasing the dataset. While one can reference `data/labeled_ableism_complete_dataset_filtered.csv` for reproducibility, researchers can request access by contacting the authors in the paper.

## License

This project is licensed under the MIT License.

## Citation


If you use this project in your research, please consider citing our work:

```bibtex
@article{phutane2025ableist,
  title={ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios},
  author={Phutane, Mahika and Jung, Hayoung and Kim, Matthew and Mitra, Tanushree and Vashistha, Aditya},
  journal={arXiv preprint arXiv:2510.10998},
  year={2025}
}
```
