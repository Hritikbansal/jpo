import os
import csv
import json
import time 
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from comparison_prompts import PROMPT_SINGLE, PROMPT_PAIR

parser = argparse.ArgumentParser()

parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo-0125'], default='gpt-3.5-turbo-0125')
parser.add_argument('--mode', choices = ['single', 'pair'], default = 'single')
parser.add_argument('--input_data', type = str, default = 'output generations')
parser.add_argument('--output_data', type = str, default = 'feedback output')

args = parser.parse_args()

def get_feedback(feedback_1, feedback_2):
    if '(a)' in feedback_1 and '(b)' in feedback_2:
        feedback = 0
    elif '(b)' in feedback_1 and '(a)' in feedback_2:
        feedback = 1
    ## 2 means equal
    elif '(a)' in feedback_1 and '(a)' in feedback_2:
        feedback = 2
    elif '(b)' in feedback_1 and '(b)' in feedback_2:
        feedback = 2
    else:
        feedback = None
    return feedback

def main():

    with open(args.input_data, 'r') as f:
        input_data = list(f)
    
    delimiter = "<SEP><SEP><SEP><SEP>"

    if os.path.exists(args.output_data):
        with open(args.output_data, 'r') as f:
            master = list(f)
        if args.mode == 'single':
            instructions_completed = set([eval(x)['instruction'] for x in master])
        else:
            instructions_completed = set([delimiter.join([eval(x)['instruction_0'], eval(x)['instruction_1']]) for x in master])
    else:
        instructions_completed = set()
    
    for j in tqdm(range(len(input_data))):
        if j != 0 and j % 50 == 0:
            time.sleep(5)

        if args.mode == "single":
            instance = eval(input_data[j])
            instruction = instance['instruction']
            response_0 = instance['response_0']
            response_1 = instance['response_1']
            instruction_prompt = f"{instruction}"
            if not (instruction in instructions_completed):
                try:
                    prompt_AB = PROMPT_SINGLE.format(instruction=instruction_prompt, output_1=response_0, output_2=response_1)
                    messages = [{"role": "user", "content": prompt_AB}]
                    completion = openai.ChatCompletion.create(
                        model = args.gpt_version, 
                        messages = messages)
                    feedback_1 = completion['choices'][0]['message']['content']
                    prompt_BA = PROMPT_SINGLE.format(instruction=instruction_prompt, output_1=response_1, output_2=response_0)
                    messages = [{"role": "user", "content": prompt_BA}]
                    completion = openai.ChatCompletion.create(
                        model = args.gpt_version, 
                        messages = messages)
                    feedback_2 = completion['choices'][0]['message']['content']
                    feedback = get_feedback(feedback_1, feedback_2)
                    print(feedback)
                    if feedback == None:
                        continue
                    instance = {'instruction': instruction, 'response_0': response_0, 'response_1': response_1, 'feedback': feedback}
                    with open(args.output_data, 'a') as f:
                        strg = json.dumps(instance)
                        f.write(strg + "\n")
                except:
                    print('sleeping')
                    time.sleep(5)
        elif args.mode == 'pair':
            instance = eval(input_data[j])
            instruction_0 = instance['instruction_0']
            instruction_1 = instance['instruction_1']
            response_0 = instance['response_0']
            response_1 = instance['response_1']
            instruction_prompt_0 = f"{instruction_0}"
            instruction_prompt_1 = f"{instruction_1}"
            key = delimiter.join([instruction_0, instruction_1])
            if not (key in instructions_completed):
                try:
                    prompt_AB = PROMPT_PAIR.format(instruction_1=instruction_prompt_0, instruction_2=instruction_prompt_1, output_1=response_0, output_2=response_1)
                    messages = [{"role": "user", "content": prompt_AB}]
                    completion = openai.ChatCompletion.create(
                        model = args.gpt_version, 
                        messages = messages)
                    feedback_1 = completion['choices'][0]['message']['content']
                    prompt_BA = PROMPT_PAIR.format(instruction_1=instruction_prompt_1, instruction_2=instruction_prompt_0, output_1=response_1, output_2=response_0)
                    messages = [{"role": "user", "content": prompt_BA}]
                    completion = openai.ChatCompletion.create(
                        model = args.gpt_version, 
                        messages = messages)
                    feedback_2 = completion['choices'][0]['message']['content']
                    feedback = get_feedback(feedback_1, feedback_2)
                    print(feedback)
                    if feedback == None:
                        continue
                    instance = {'instruction_0': instruction_0, 'response_0': response_0, 'instruction_1': instruction_1, 'response_1': response_1, 'feedback': feedback}
                    with open(args.output_data, 'a') as f:
                        strg = json.dumps(instance)
                        f.write(strg + "\n")
                except:
                    print('sleeping')
                    time.sleep(5)  
                    
if __name__ == '__main__':
    main()