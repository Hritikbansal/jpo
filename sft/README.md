# Supervised Finetuning Mistral-7B

This repository contains the code necessary for fine-tuning the Mistral-7B model. To prepare for fine-tuning, export the environment variable TRAIN_PATH and VALIDATION_PATH with `train_sft.jsonl` and `val_sft.jsonl`. Alternatively, can be passed in the command line during training as shown in following sections.

**Important Notice**: Supervised Fine Tuning is exclusively designed for the full precision fine-tuning of the Mistral model. It is not suitable for QLORA or other fine-tuning methodologies.

# How to run
- Acquire a Hugging Face authentication token by visiting [here](https://huggingface.co/settings/tokens).
- Export the environment variable with the obtained token. Alternatively, can be passed in the command line as shown in following section -
- Change the prefix prompt according to the dataset [here](https://github.com/Hritikbansal/jpo/blob/main/sft/core/supervised_dataset.py).

Run training code:

```
TRAIN_PATH=../data/tldr/train_sft.jsonl VALIDATION_PATH=../data/tldr/val_sft.jsonl HF_HOME=/data/gbhatt2/ HF_TOKEN=[your auth token here] CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc-per-node=2 train.py
```

# Recommendations for Effective Fine-Tuning

- When operating with a smaller batch size, it is advisable to reduce the learning rate accordinµgly.
- Adjustments to gradient clipping and weight decay were not necessary in our experience, but this may vary depending on your specific context.
- For optimal results, a dataset containing more than 1,000 samples is recommended.
- Our tests involved running the fine-tuning process for 3 epochs on a dataset of 40,000 samples. Further experimentation with the number of epochs is encouraged, as improvements were observed beyond this point.
- To accurately assess whether your model is progressing, becoming overfitted, or declining in performance, incorporate evaluation mechanisms on downstream validation data.
- Regarding the Fully Sharded Data Parallel (FSDP) option, use `backward_prefetch=BackwardPrefetch.BACKWARD_PRE` if sufficient GPU memory is available. Alternatively, `backward_prefetch=BackwardPrefetch.BACKWARD_POST` may be utilized. Note that to prevent out-of-memory (OOM) errors, this option was set to None in our configurations.
