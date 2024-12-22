import torch 
import json
import argparse
import transformers 
from tqdm import tqdm 
from peft import PeftModel, LoraConfig

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type = str, help = 'trained model ckpt')
parser.add_argument('--test_file', type = str, help = 'test file')
parser.add_argument('--output_file', type = str, default = 'output file')
parser.add_argument('--temp', type = float, default = 0.001)

args = parser.parse_args()

model_path = args.model_path
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map = "auto", low_cpu_mem_usage=True)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

def generate_samples(temp):
    with open(args.test_file, 'r') as f:
        data = list(f)
        
    for example in tqdm(data):
            example = eval(example)
            new_dict = {}
            if 'helpful' in args.model_path:
                prompt = f"### Instructions: {example['instruction']}\n\n### Response:"
            else:
                prompt = f"### Instructions: You are given a social media post, summarize it accordingly. \nPost: {example['instruction']}\n\n### Response:"
            inputs = tokenizer(prompt, return_tensors="pt")
            output_texts = model.generate(inputs=inputs.input_ids.to("cuda"), max_new_tokens = 400, num_return_sequences=1, temperature = temp, do_sample=True)
            output_texts = tokenizer.batch_decode(output_texts, skip_special_tokens=True)
            new_dict['instruction'] = prompt
            new_dict['outputs'] = output_texts[0].split('### Response:')[-1]            
            
            with open(f'{args.output_file[:-5]}_{temp}.json', 'a') as f:
                strg = json.dumps(new_dict)
                f.write(strg + "\n")

def main():
    # temperatures = [0.001]
    # temperatures = [0.001, 0.5, 1.0]
    # for temp in temperatures:
        # generate_samples(temp)
    generate_samples(args.temp)
    
           
if __name__ == '__main__':
    main()