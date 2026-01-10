from openai import OpenAI, AsyncOpenAI
from google import genai
from google.genai import types
from googleapiclient import discovery
from anthropic import Anthropic, AsyncAnthropic
from transformers import pipeline
import torch
import pandas as pd

import json
from dotenv import load_dotenv
import os
import asyncio
from typing import List, Dict, Any

class LLMInterface:
    huggingface_models = {}
    
    def __init__(self, company, model):
        self.api_key = ""
        self.company = company
        self.model = model
        self.pipe = None  # Store pipeline to manage memory for huggingface models
        self.temperature = 0.7
        self.max_tokens = 1024

    def perspective(self, input, **kwargs):
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        )

        analyze_request = {
            "comment": {
                "text": input
            },
            "languages": ["en"],
            "requestedAttributes": {
                attr.strip().upper().replace(" ", "_"): {}
                for attr in kwargs.get("attributes", ["TOXICITY", "SEVERE TOXICITY", "IDENTITY ATTACK", "INSULT", "THREAT", "PROFANITY"])
            }
        }

        response = client.comments().analyze(body=analyze_request).execute()
        return json.dumps(response, indent=2)

    def openai_moderation(self, input, **kwargs):
        client = OpenAI(api_key=self.api_key)
        model_name = kwargs.get("model", self.model) or "omni-moderation-latest"
        response = client.moderations.create(
            model=model_name,
            input=input,
        )

        result = response.results[0]
        return {
            "flagged": getattr(result, "flagged", None),
            "categories": getattr(result, "categories", None),
            "category_scores": getattr(result, "category_scores", None),
            "model": model_name,
        }        

    def huggingface(self, input, **kwargs):       
        messages = [
            {"role": "system", "content": kwargs.get("instructions", "")},
            {"role": "user", "content": input},
        ]
        piped_response = self.pipe(messages, max_new_tokens=kwargs.get("max_tokens", self.max_tokens), do_sample=True, temperature=kwargs.get("temperature", self.temperature))
        return piped_response[0]["generated_text"][-1]["content"]
    
    def run(self, input, **kwargs):
        load_dotenv()
        match self.company:
            case "openai_moderation":
                self.api_key = os.getenv("OPENAI_KEY")
                if not self.model:
                    self.model = "omni-moderation-latest"
                return self.openai_moderation(input, **kwargs)
            case "perspective":
                self.api_key = os.getenv("PERSPECTIVE_KEY")
                return self.perspective(input, **kwargs)
            case _:
                raise ValueError("Invalid company for synchronous run. Supported: 'openai_moderation', 'perspective'.")
                
    async def gpt_async(self, input, **kwargs):
        client = AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": kwargs.get("instructions", "")},
                {"role": "user", "content": input},
            ],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        return response.choices[0].message.content

    async def google_async(self, input, **kwargs):
        client = genai.Client(api_key=self.api_key)
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=input,
            config=types.GenerateContentConfig(
                system_instruction=kwargs.get("instructions", ""),
                temperature=kwargs.get("temperature", self.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                response_modalities=["TEXT"],
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )

        return response.text
    
    async def anthropic_async(self, input, **kwargs):
        client = AsyncAnthropic(api_key=self.api_key)

        response = await client.messages.create(
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            model=self.model,
            system=kwargs.get("instructions", ""),
            messages=[{"role": "user", "content": input}],
            temperature=kwargs.get("temperature", self.temperature),
        )

        return response.content[0].text

    async def deepseek_async(self, input, **kwargs):
        client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": kwargs.get("instructions", "")},
                {"role": "user", "content": input},
            ],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=False
        )

        return response.choices[0].message.content

    async def run_async(self, input, **kwargs):
        load_dotenv()

        match self.company:
            case "openai":
                self.api_key = os.getenv("OPENAI_KEY")
                return await self.gpt_async(input, **kwargs)
            case "google":
                self.api_key = os.getenv("GOOGLE_KEY")
                return await self.google_async(input, **kwargs)
            case "anthropic":
                self.api_key = os.getenv("ANTHROPIC_KEY")
                return await self.anthropic_async(input, **kwargs)
            case "deepseek":
                self.api_key = os.getenv("DEEPSEEK_KEY")
                return await self.deepseek_async(input, **kwargs)
            case "huggingface":
                self.api_key = os.getenv("HF_TOKEN")
                return self.huggingface(input, **kwargs)
            case "perspective":
                self.api_key = os.getenv("PERSPECTIVE_KEY")
                return self.perspective(input, **kwargs)
            case _:
                print("Invalid company")
                exit()

    def run_perspective_csv(self, input_csv_path, output_csv_path, **kwargs):
        df = pd.read_csv(input_csv_path)
        for index, row in df.iterrows():
            response = self.perspective(row['Response'], **kwargs)
            df.at[index, 'Response'] = json.dumps(response, indent=2)
        df.to_csv(output_csv_path, index=False)

    async def run_multiple_llms_async(self, llm_configs: List[Dict[str, Any]], input: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Run multiple LLMs asynchronously with the same input. Note: Huggingface and perspective are synchronous.
        
        Args:
            llm_configs: List of dictionaries containing 'company' and 'model' keys
                        Example: [{'company': 'openai', 'model': 'gpt-4'}, 
                                 {'company': 'anthropic', 'model': 'claude-3-sonnet-20240229'}]
            input: The input text to send to all LLMs
            **kwargs: Additional parameters like instructions, temperature, max_tokens
            
        Returns:
            List of dictionaries containing 'company', 'model', 'response', and 'error' (if any)
        """
        tasks = []
        llm_instances = []
        
        for config in llm_configs:
            llm = LLMInterface(config['company'], config['model'])
            llm_instances.append((llm, config))
            task = asyncio.create_task(llm.run_async(input, **kwargs))
            tasks.append(task)
        
        results = []
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (result, (llm, config)) in enumerate(zip(completed_tasks, llm_instances)):
            if isinstance(result, Exception):
                results.append({
                    'company': config['company'],
                    'model': config['model'],
                    'response': None,
                    'error': str(result)
                })
            else:
                results.append({
                    'company': config['company'],
                    'model': config['model'],
                    'response': result,
                    'error': None
                })
        
        return results
