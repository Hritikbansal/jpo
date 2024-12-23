
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from custom_jpo_trainer import CustomJpoTrainer
# from trl import DPOTrainer
from utils import *

cache_dir = "your cache directory here"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ["HF_TOKEN"] = "your_token_here"
# os.environ["WANDB_DISABLED"] = "true"

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="mistral_7b_lr_1.5e-6_sft",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name: Optional[str] = field(default="train_pref.jsonl", metadata={"help": "dataset_name"})
    eval_dataset_name: Optional[str] = field(default="val_pref.jsonl", metadata={"help": "dataset_name"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=300, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=300, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # Dataset property, joint modeling / augmented datset 
    joint_distribution: Optional[bool] = field(default=False, metadata={"help": "Trains on joint distribution as objective instead of conditional"})


    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 100 samples"})
    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        #low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Training and validation datasets.

    train_dataset = load_dataset("json", data_files =script_args.dataset_name, split="train")
    train_dataset = train_dataset.map(
            return_prompt_and_responses_augmented,
            batched=True,
            num_proc=24,
        )

    train_dataset = filter_long_sequences(train_dataset, script_args.max_length)

    
    # 3. Load evaluation dataset
    eval_dataset = load_dataset("json", data_files = script_args.eval_dataset_name, split="train")
    eval_dataset = eval_dataset.map(
            return_prompt_and_responses_augmented,
            batched=True,
            num_proc=24,
        )

    eval_dataset = filter_long_sequences(eval_dataset, script_args.max_length)

    if len(script_args.output_dir) <= 1:
        model_name = script_args.model_name_or_path.replace('../', '')
        model_name = model_name.replace("..","").replace("/", "_")
        dataset_name = script_args.dataset_name.replace('/','_')
        batch = script_args.per_device_train_batch_size
        script_args.output_dir = f"{model_name}_{dataset_name}_lr{script_args.learning_rate}_{script_args.lr_scheduler_type}_b{batch}_step{script_args.max_steps}"
        print('output', script_args.output_dir)
        
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="",
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    run_name = f"name_{script_args.output_dir}_{script_args.model_name_or_path}_bsz={script_args.per_device_train_batch_size}_joint={script_args.joint_distribution}"
    if script_args.report_to == 'wandb':   
        import wandb 
        wandb.init(project="project", config=vars(script_args), name=run_name)

    # 5. initialize the Jpo trainer
    jpo_trainer = CustomJpoTrainer(
        model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        joint_distribution = script_args.joint_distribution,
    )

    # 6. train
    # jpo_trainer.train(resume_from_checkpoint=True)
    jpo_trainer.train()
    jpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    jpo_trainer.model.save_pretrained(output_dir)


"""
    CUDA_VISIBLE_DEVICES=0 python vanilla_jpo_trainer.py --dataset_name ../data/tldr_data/openai_tldr_unique_dpo_format.jsonl --eval_dataset_name ../data/tldr_data/val_dpo_format.jsonl --output_dir /local2/hbansal/pref_augment/
"""
