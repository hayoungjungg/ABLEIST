import tiktoken
from utils import GPTRequests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

"""
Constructs the provided prompt given the DataFrame containing the responses.

If providing zero-shot prompt, please pass in None value for the few_shot parameter.
"""
def create_prompts(df, prompt, persona, metric, metric_definition, few_shot):
    id_to_prompt = dict()
    for i, row in df.iterrows():
        new_prompt = str(prompt)
        
        id = row['index']
        response = row['Response']
        annotation_label = row[metric + ' (Final)']
         
        new_prompt = new_prompt.replace('[METRIC]', metric)
        new_prompt = new_prompt.replace('[DEFINITION]', metric_definition)
        new_prompt = new_prompt.replace('[CONVERSATION]', response)

        if few_shot:
            new_prompt = new_prompt.replace('[FEW-SHOT]', few_shot)
        id_to_prompt[id] = ([{"role": "system", "content": persona}, {"role": "user", "content": f"{new_prompt}"}], annotation_label)
        
    total_token, avg_tokens = average_input_tokens(id_to_prompt)
    print("The total input tokens: " + str(total_token))
    print("The average input tokens: " + str(avg_tokens))
        
    return id_to_prompt

"""
Constructs the provided prompt given the DataFrame containing the responses.

If providing zero-shot prompt, please pass in None value for the few_shot parameter.
"""
def create_single_prompt(response, prompt, persona, metric, metric_definition, few_shot):
    new_prompt = str(prompt)

    new_prompt = new_prompt.replace('[METRIC]', metric)
    new_prompt = new_prompt.replace('[DEFINITION]', metric_definition)
    new_prompt = new_prompt.replace('[CONVERSATION]', response)

    if few_shot:
        new_prompt = new_prompt.replace('[FEW-SHOT]', few_shot)
    return ([{"role": "system", "content": persona}, {"role": "user", "content": f"{new_prompt}"}])
        


def create_prompts_all_data(df, prompt, persona, metric, metric_definition, few_shot):
    id_to_prompt = dict()
    need_labels = df[df[metric].isna()]
    print("Need Labels: " + str(need_labels.shape[0]))
    for i, row in need_labels.iterrows():
        new_prompt = str(prompt)
        
        id = row['index']
        response = row['Response']
         
        new_prompt = new_prompt.replace('[METRIC]', metric)
        new_prompt = new_prompt.replace('[DEFINITION]', metric_definition)
        new_prompt = new_prompt.replace('[CONVERSATION]', response)

        if few_shot:
            new_prompt = new_prompt.replace('[FEW-SHOT]', few_shot)
        id_to_prompt[id] = ([{"role": "system", "content": persona}, {"role": "user", "content": f"{new_prompt}"}])
        
    total_token, avg_tokens = average_input_tokens(id_to_prompt)
    print("The total input tokens: " + str(total_token))
    print("The average input tokens: " + str(avg_tokens))
        
    return id_to_prompt



"""
Calculates the average input tokens to estimate cost and decide which model to use
"""
def average_input_tokens(prompts, model_name="gpt-3.5-turbo"):
    # Load the encoding for the model
    encoding = tiktoken.encoding_for_model(model_name)
    
    # Calculate the number of tokens for each prompt
    #print(prompts.keys()) 
    #token_counts = [len(encoding.encode(prompt[1]['content'])) for prompt in prompts.values()]
    token_counts = [len(encoding.encode(prompt[0][1]['content'])) for prompt in prompts.values()]
    
    # Calculate the average token count
    total_token = sum(token_counts)
    avg_tokens = total_token / len(prompts) if prompts else 0
    
    return total_token, avg_tokens

"""
Function to compute the performance results compared to the ground-truth labels

Input:
- id_to_output: Dictionary containing the prediction results
- chat_completion_bool: True if Chat Completion object, False otherwise
- list_id_exclude: list of video ids to exclude from the result computation
"""
def compute_results(id_to_output, chat_completion_bool, list_id_exclude):
    # extracting the labels and organizing into list
    predictions = []
    ground_truths = []
    for id, (output, gold) in id_to_output.items():
        if id not in list_id_exclude:
            if type(output) == int:
                predictions.append(output)
            else:
                print(id)
                predictions.append(GPTRequests.extract_label(output, chat_completion_bool))
            ground_truths.append(gold)

    # computing accuracy, macro and weighted F1-scores, precision and recall
    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average='macro')
    f1_weighted = f1_score(ground_truths, predictions, average='weighted')

    # Compute precision and recall (macro average)
    precision = precision_score(ground_truths, predictions, average='macro')
    recall = recall_score(ground_truths, predictions, average='macro')

    # Generate classification report
    class_report = classification_report(ground_truths, predictions)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(ground_truths, predictions)

    # Print metrics
    print("Accuracy:", round(accuracy, 3))
    print("F1-Score (Macro):", round(f1_macro, 3))
    print("F1-Score (Weighted):", round(f1_weighted, 3))
    print("Precision (Macro):", round(precision, 3))
    print("Recall (Macro):", round(recall, 3))

"""
Function to compute the performance results compared to the ground-truth labels

Input:
- id_to_output: Dictionary containing the prediction results
- chat_completion_bool: True if Chat Completion object, False otherwise
- list_id_exclude: list of video ids to exclude from the result computation
"""
def compute_results_fetch_from_source(id_to_output, chat_completion_bool, eval_set, list_id_exclude, metric):
    # extracting the labels and organizing into list
    predictions = []
    ground_truths = []
    for id, output in id_to_output.items():
        if int(id) not in list_id_exclude:
            if type(output) == int:
                predictions.append(output)
            else:
                pred = GPTRequests.extract_label(output, chat_completion_bool)
                predictions.append(pred)
            selected_row = eval_set.loc[eval_set['index'] == int(id)]
            gold = selected_row[metric + ' (Final)'].iloc[0]
            
            ground_truths.append(gold)
            
    print(len(predictions))
    print(len(ground_truths))

    # computing accuracy, macro and weighted F1-scores, precision and recall
    try:
        accuracy = accuracy_score(ground_truths, predictions)
    except Exception as e:
        print(e)
        print(ground_truths)
    print(predictions)
    f1_macro = f1_score(ground_truths, predictions, average='macro')
    f1_weighted = f1_score(ground_truths, predictions, average='weighted')

    # Compute precision and recall (macro average)
    precision = precision_score(ground_truths, predictions, average='macro')
    recall = recall_score(ground_truths, predictions, average='macro')

    # Generate classification report
    class_report = classification_report(ground_truths, predictions)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(ground_truths, predictions)

    # Print metrics
    print("Accuracy:", round(accuracy, 3))
    print("F1-Score (Macro):", round(f1_macro, 3))
    print("F1-Score (Weighted):", round(f1_weighted, 3))
    print("Precision (Macro):", round(precision, 3))
    print("Recall (Macro):", round(recall, 3))
