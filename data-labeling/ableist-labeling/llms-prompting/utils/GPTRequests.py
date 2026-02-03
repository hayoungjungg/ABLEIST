import os
import json
from openai import OpenAI
from dotenv import dotenv_values
from collections import defaultdict
import typing_extensions as typing
import tiktoken
import re

def get_response_claude(client, model_name, prompt, temperature, reasoning_effort_):
    ct, num_tokens, num_completion_tokens, num_prompt_tokens = 0, 0, 0, 0
    response = ""
    
    failed = False
    while ct < 1:
        ct += 1
        try:            
            response = client.messages.create(
                    model=model_name,
                    messages=[prompt[1]], 
                    temperature=temperature,  # Control the randomness of the output
                    max_tokens=2000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                )
           
            num_tokens += 0
            num_completion_tokens += getattr(response.usage, "output_tokens", 0)
            num_prompt_tokens += getattr(response.usage, "input_tokens", 0)
            
            # Get the thinking string (first ThinkingBlock)
            thinking_text = next(
                (b.thinking for b in response.content if getattr(b, "type", None) == "thinking"),
                None
            )

            # Get the text block string (first TextBlock)
            text_output = next(
                (b.text for b in response.content if getattr(b, "type", None) == "text"),
                None
            )
            response = text_output + '\n\n\n' + thinking_text
            
            # checking well-formed json content
            #output = response.choices[0].message.content
            return response, num_tokens, num_completion_tokens, num_prompt_tokens #.choices[0].message.content

        except Exception as e:
            print("Error")
            print(e)
            print(prompt)
            continue
            #logging.error(traceback.format_exc())
    return response, num_tokens, num_completion_tokens, num_prompt_tokens


def get_response(client, model_name, prompt, temperature, reasoning_effort_):
    ct, num_tokens, num_completion_tokens, num_prompt_tokens = 0, 0, 0, 0
    response = ""
    
    failed = False
    while ct < 1:
        ct += 1
        try:
            response = ""
            if model_name == 'gpt-5-chat-latest' or 'claude' in model_name:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=prompt, 
                    temperature=temperature,  # Control the randomness of the output
                    response_format={"type": "json_object"},  # Ensure output is in JSON format
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=prompt, 
                    response_format={"type": "json_object"},  # Ensure output is in JSON format
                    reasoning_effort=reasoning_effort_
                )
            num_tokens += response.usage.total_tokens
            num_completion_tokens += response.usage.completion_tokens
            num_prompt_tokens += response.usage.prompt_tokens
            
            # checking well-formed json content
            output = response.choices[0].message.content
            try:
                json_output = json.loads(output)
                
                if failed:
                    print("Succeeded Retry. Continuing")
                
                return response, num_tokens, num_completion_tokens, num_prompt_tokens #.choices[0].message.content
            except json.JSONDecodeError as json_error:
                #print(f"Malformed JSON content: {json_error}. Retrying ({ct}/3)...")
                failed = True
                continue  # Retry the request
        except Exception as e:
            print("Error")
            print(e)
            print(prompt)
            continue
            #logging.error(traceback.format_exc())
    return response, num_tokens, num_completion_tokens, num_prompt_tokens

"""
Helper function to iteratively evaluate prompts through the LLM (e.g., model_name) at the given temperature.

Returns the token usage and dictionary (id to output mapping)
"""
def evaluate_prompts_claude(client, provided_prompt, model_name, temperature, reasoning_effort):
    id_to_output = defaultdict()

    # intermediate variables
    total_completion_token = 0
    total_prompt_token = 0

    # iterating through each prompt into the LLM
    for i, (index, prompt) in enumerate(provided_prompt.items()):
        if i % 10 ==0:
            print(i)
            
        extracted = prompt
        if isinstance(prompt, tuple):
            extracted = prompt[0]
        
        response, num_tokens, num_completion_tokens, num_prompt_tokens = get_response_claude(client, model_name, extracted, temperature, reasoning_effort)
        
        # saving the outputs
        total_completion_token += num_completion_tokens
        total_prompt_token += num_prompt_tokens
        id_to_output[index] = response

    print("Total Number of Input Token: " + str(total_prompt_token))
    print("Total Number of Output Token: " + str(total_completion_token))
    return id_to_output, total_completion_token, total_prompt_token



"""
Helper function to iteratively evaluate prompts through the LLM (e.g., model_name) at the given temperature.

Returns the token usage and dictionary (id to output mapping)
"""
def evaluate_prompts(client, provided_prompt, model_name, temperature, reasoning_effort):
    id_to_output = defaultdict()

    # intermediate variables
    total_completion_token = 0
    total_prompt_token = 0

    # iterating through each prompt into the LLM
    for i, (index, prompt) in enumerate(provided_prompt.items()):
        if i % 10 ==0:
            print(i)
            
        extracted = prompt
        if isinstance(prompt, tuple):
            extracted = prompt[0]
        
        response, num_tokens, num_completion_tokens, num_prompt_tokens = get_response(client, model_name, extracted, temperature, reasoning_effort)
        
        # saving the outputs
        total_completion_token += num_completion_tokens
        total_prompt_token += num_prompt_tokens
        id_to_output[index] = response

    print("Total Number of Input Token: " + str(total_prompt_token))
    print("Total Number of Output Token: " + str(total_completion_token))
    return id_to_output, total_completion_token, total_prompt_token

def extract_binary_label(text: str):
    """
    Return 0 or 1 if a label is found in a messy string, else None.
    Covers JSON-style, markdown-ish, array-wrapped labels, code fences,
    and double-encoded strings.
    """
    # 1) Try to unescape if it's a JSON string literal (up to twice for double-encoding)
    for _ in range(2):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                text = decoded
            else:
                # decoded into a dict/list; stringify back for regex scan
                text = json.dumps(decoded)
        except Exception:
            break

    # 2) Strip fenced code blocks like ```json ... ```
    #    (do this twice to be safe if there are multiple)
    text = re.sub(r'```.*?\n', '', text, flags=re.DOTALL)
    text = text.replace('```', '')

    # 3) Primary patterns (ordered: array first, then scalar)
    patterns = [
        r'(?i)"label"\s*:\s*\[\s*([01])\s*\]',          # "LABEL": [1]
        r'(?i)"label"\s*:\s*([01])',                    # "LABEL": 1
        r'(?i)\*{0,2}\s*label\s*\*{0,2}\s*:\s*\[\s*([01])\s*\]',  # **LABEL**: [1]
        r'(?i)\*{0,2}\s*label\s*\*{0,2}\s*:\s*([01])',  # **LABEL**: 1
        r'(?i)\blabel\b\s*:\s*\[\s*([01])\s*\]',        # LABEL: [1]
        r'(?i)\blabel\b\s*:\s*([01])',                  # LABEL: 1
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))

    # 4) Fallback: if we at least see "label" somewhere, try nearest [0]/[1]
    if re.search(r'(?i)\blabel\b', text):
        m = re.search(r'\[\s*([01])\s*\]', text)
        if m:
            return int(m.group(1))

    # 5) Last-resort fallback: any [0]/[1] at all in the string
    m = re.search(r'\[\s*([01])\s*\]', text)
    if m:
        return int(m.group(1))

    return None

"""
Given the completion output, extracts the label from the generation.

By default, add the chat completion object returned by GPT-4, but if already extracted,
add the chat content in the completion_output parameter
"""
def extract_label(completion_output, extracted=False):
    # extracting the content & converting the string to json
    output = ""
    if extracted:
        output = completion_output.choices[0].message.content  
    else:
        output = completion_output
    
    json_output = ""
    try:
        json_output = json.loads(output)    
    except:
        #print("parse error")
        #print(output)
        label = extract_binary_label(output)
        #print(label)
        return label
    
    # extracting the label
    try:
        if isinstance(json_output, list):
            json_output = json_output[0]
        
        label = str(json_output['LABEL'])
    except Exception as e:
        print("label extraction error")
        print(json_output)
        print(e)
    
    # standardizing the label
    if '0' in label or label == 0:
        return 0
    elif '1' in label or label == 1:
        return 1
    else:
        print("Error parsing the label")
        print(label)
        return None
    
"""
Extracts the content from the evaluation_dictionary and saves into a file called
file_name
"""
def extract_and_save_output(file_name, evaluation_dictionary):
    # extracting the content from the chat completion object
    cleaned_dictionary_to_save = dict()
    for vid, chat_completion in evaluation_dictionary.items():
        try:
            cleaned_dictionary_to_save[vid] = chat_completion.choices[0].message.content  
        except Exception as e:
            print(e)
            print(chat_completion)
            print(vid)
            cleaned_dictionary_to_save[vid] = chat_completion

    #print(cleaned_dictionary_to_save[165])
    save_output(file_name, cleaned_dictionary_to_save)
    
def extract_output_only(evaluation_dictionary):
    # extracting the content from the chat completion object
    cleaned_dictionary_to_save = dict()
    for vid, chat_completion in evaluation_dictionary.items():
        try:
            cleaned_dictionary_to_save[str(vid)] = chat_completion[0].choices[0].message.content  
        except Exception as e:
            print(e)
            print(chat_completion)
            print(vid)
            cleaned_dictionary_to_save[vid] = chat_completion
    return cleaned_dictionary_to_save
        
"""
Extracts the content from the evaluation_dictionary and saves into a file called
file_name
"""
def save_output(file_name, evaluation_dictionary):
    # saving the file
    with open(file_name, 'w') as json_file:
        json.dump(evaluation_dictionary, json_file, indent=4)

"""
Loads the content back 
"""
def load_output(file_name):
    # reading the file
    with open(file_name, 'r') as json_file:
        evaluation_dictionary = json.load(json_file)
    return evaluation_dictionary
