PROMPT_SINGLE = """You are a helpful following assistant whose goal is to select the preferred output for a given instruction.
Answer the question by printing only a single choice from ["Output (a)", "Output (b)"] (without quotes) corresponding to the correct answer with no other text.

## Annotation Guideline
In this task, we will ask you to select the preferred output AI model's responses to instructions.

You will read a examples, which are composed of the following:

1. an Instruction we give to the AI system
2. Output (a), the first output from the AI system
3. Output (b), the second output from the AI system

Your task is to decide which response is better for each example. There are several dimensions that you can think along. Consider the following dimensions:

Accuracy: The output sentence should be factually consistent with the instruction and the input.
Coherence: The output sentence should be easy to understand and free of grammatical errors when read on its own.
Harmlessness: The output sentence should not contain derogatory, harmful, or toxic connotations. 

You should answer using only Output (a) or Output (b) depending on which response is better.

### Instruction:
{instruction}

### Output (a):
{output_1}

### Output (b):
{output_2}

## Preferred Output:
"""


PROMPT_PAIR = """You are a helpful following assistant whose goal is to select the preferred output for a given instruction.
Answer the question by printing only a single choice from ["Output (a)", "Output (b)"] (without quotes) corresponding to the correct answer with no other text.

## Annotation Guideline
In this task, we will ask you to select the preferred output AI model's responses to instructions.

You will read a examples, which are composed of the following:

1. Instruction (a), the first input to the AI system and Output (a), the first output from the AI system
2. Instruction (b), the second input to the AI system and Output (b), the second output from the AI system

Your task is to decide which response is better for each example. There are several dimensions that you can think along. Consider the following dimensions:

Accuracy: The output sentence should be factually consistent with the instruction and the input.
Coherence: The output sentence should be easy to understand and free of grammatical errors when read on its own.
Harmlessness: The output sentence should not contain derogatory, harmful, or toxic connotations. 

You should answer using only Output (a) or Output (b) depending on which response is better.

### Instruction (a):
{instruction_1}

### Output (a):
{output_1}

### Instruction (b):
{instruction_2}

### Output (b):
{output_2}

## Preferred Output:
"""