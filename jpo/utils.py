from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

def return_prompt_and_responses_augmented(samples) -> Dict[str, str]:
    return {
        "prompt_chosen": [fmt_prompt(question) for question in samples["i_chosen"]],
        "response_chosen": samples["r_chosen"],
        "prompt_rejected": [fmt_prompt(question) for question in samples["i_reject"]],
        "response_rejected": samples["r_reject"],
    }

def fmt_prompt(prompt:str):
    ## uncomment below for the summarization dataset
    # return f"### Instructions: You are given a social media post, summarize it accordingly. \nPost: {prompt}\n\n### Response:"
    return f"### Instructions: {prompt}\n\n### Response:"

def filter_long_sequences(dataset, max_length:int = 512):
    dataset = dataset.filter(
        lambda x: len(x["prompt_chosen"]) + len(x["response_chosen"]) <= max_length
        and len(x["prompt_rejected"]) + len(x["response_rejected"]) <= max_length
    )
    return dataset
