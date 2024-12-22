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

parser.add_argument('--gpt_version', default='gpt-3.5-turbo-0125')
parser.add_argument('--input_data', type = str, help = 'output generations')
parser.add_argument('--model_name', type = str, help = 'name of the model')
parser.add_argument('--test_data', type = str, default = 'openai_tldr_test.json')
parser.add_argument('--master_data', type = str, help = 'master file which contains the GPT feedback')

args = parser.parse_args()

def get_feedback(feedback_1, feedback_2):
    if '(a)' in feedback_1 and '(b)' in feedback_2:
        feedback = 'gold'
    elif '(b)' in feedback_1 and '(a)' in feedback_2:
        feedback = 'model'
    elif '(a)' in feedback_1 and '(a)' in feedback_2:
        feedback = 'equal'
    elif '(b)' in feedback_1 and '(b)' in feedback_2:
        feedback = 'equal'
    else:
        feedback = None
    return feedback



def main():

    with open(args.input_data, 'r') as f:
        input_data = list(f)
    
    if os.path.exists(args.master_data):
        with open(args.master_data, 'r') as f:
            master = json.load(f)
    else:
        master = {}

    with open(args.test_data, 'r') as f:
        test_data = list(f)

    method = args.model_name
    if not (method in master):
        master[method] = {}

    delimiter = "<SEP><SEP><SEP><SEP>"

    score = 0
    total = 0
    
    for j in tqdm(range(len(input_data[:500]))):
        if j != 0 and j % 50 == 0:
            time.sleep(5)

        instance = eval(input_data[j])
        instruction = instance['instruction']
        model_response = instance['outputs']

        testinstance = eval(test_data[j])
        gold = testinstance['gold_response']

        key = delimiter.join([instruction, gold, model_response])
        feedback = None
        if not (key in master[method]):
            prompt_AB = PROMPT_SINGLE.format(instruction=instruction, output_1=gold, output_2=model_response)
            messages = [{"role": "user", "content": prompt_AB}]
            completion = openai.ChatCompletion.create(
                model = args.gpt_version, 
                messages = messages)
            feedback_1 = completion['choices'][0]['message']['content']
            prompt_BA = PROMPT_SINGLE.format(instruction=instruction, output_1=model_response, output_2=gold)
            messages = [{"role": "user", "content": prompt_BA}]
            completion = openai.ChatCompletion.create(
                model = args.gpt_version, 
                messages = messages)
            feedback_2 = completion['choices'][0]['message']['content']
            feedback = get_feedback(feedback_1, feedback_2)
            print(feedback)
            if feedback == None:
                continue
            master[method][key] = feedback
        else:
            feedback = master[method][key]
        if feedback != None:
            score += 1 if feedback == 'model' else 0.5 if feedback == 'equal' else 0
            total += 1
            print(f"Score: {score} | Total: {total} | Win-rate: {100 * score / total}")

            with open(args.master_data, 'w') as f:
                json.dump(master, f, indent = 4)

if __name__ == '__main__':
    main()