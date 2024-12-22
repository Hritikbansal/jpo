# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from accelerate.utils import is_deepspeed_available, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import DPOTrainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from datasets import Dataset
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from utils import * 
from trl.import_utils import is_peft_available, is_wandb_available
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
from accelerate import PartialState
from trl.models import PreTrainedModelWrapper, create_reference_model

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class CustomJpoTrainer(DPOTrainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper, or `"kto"` from the HALOs [report](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf).
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        joint_distribution (`bool`):
            If True, we do not apply label mask of padding.

    """

    _tag_names = ["trl", "dpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        joint_distribution: bool = False,
    ):
    
        super().__init__(
            model = model,
            ref_model = ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type = loss_type,
            args = args,
            data_collator = data_collator,
            label_pad_token_id= label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            dataset_num_proc=dataset_num_proc,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
            ref_adapter_name=ref_adapter_name,
            reference_free=reference_free)
        
        self.joint_distribution = joint_distribution # If in case we want to model the joint distribution, instead of the conditional distribution.

        print(f'prompt_length: {self.max_prompt_length}, length: {self.max_length}, beta: {self.beta}, loss_type: {self.loss_type}')

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        prompt_chosen = feature["prompt_chosen"]
        response_chosen = feature["response_chosen"]
        prompt_rejected = feature["prompt_rejected"]
        response_rejected = feature["response_rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt_chosen, str):
                raise ValueError(f"prompt should be an str but got {type(prompt_chosen)}")
            prompt_chosen_tokens = self.tokenizer(prompt_chosen, add_special_tokens=False)
            prompt_chosen_tokens = {f"prompt_{k}": v for k, v in prompt_chosen_tokens.items()}

            if not isinstance(prompt_rejected, str):
                raise ValueError(f"prompt should be an str but got {type(prompt_rejected)}")
            prompt_rejected_tokens = self.tokenizer(prompt_rejected, add_special_tokens=False)
            prompt_rejected_tokens = {f"prompt_{k}": v for k, v in prompt_rejected_tokens.items()}


            if not isinstance(response_chosen, str):
                raise ValueError(f"chosen should be an str but got {type(response_chosen)}")
            response_chosen_tokens = self.build_tokenized_answer(prompt_chosen, response_chosen)

            if not isinstance(response_rejected, str):
                raise ValueError(f"rejected should be an str but got {type(response_rejected)}")
            response_rejected_tokens = self.build_tokenized_answer(prompt_rejected, response_rejected)


            # add BOS token to head of prompt
            prompt_chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_chosen_tokens["prompt_input_ids"]
            prompt_rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_rejected_tokens["prompt_input_ids"]

            response_chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + response_chosen_tokens["prompt_input_ids"]
            response_rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + response_rejected_tokens["prompt_input_ids"]

            prompt_chosen_tokens["prompt_attention_mask"] = [1] + prompt_chosen_tokens["prompt_attention_mask"]
            prompt_rejected_tokens["prompt_attention_mask"] = [1] + prompt_rejected_tokens["prompt_attention_mask"]

            response_chosen_tokens["prompt_attention_mask"] = [1] + response_chosen_tokens["prompt_attention_mask"]
            response_rejected_tokens["prompt_attention_mask"] = [1] + response_rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            response_chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            response_chosen_tokens["attention_mask"].append(1)

            response_rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            response_rejected_tokens["attention_mask"].append(1)


            longer_response_length = max(len(response_chosen_tokens["input_ids"]), len(response_rejected_tokens["input_ids"]))

            # We will truncate for individual datapoints. 
            length_response_chosen = len(response_chosen_tokens["input_ids"])
            length_response_rejected = len(response_chosen_tokens["input_ids"])

            for answer_tokens in [prompt_chosen_tokens, response_chosen_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + length_response_chosen > self.max_length:
                    # Truncate the prompt. 
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
                    

            for answer_tokens in [prompt_rejected_tokens, response_rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + length_response_rejected > self.max_length:
                    # Truncate the prompt. 
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")


            # if that's still too long, truncate the response
            for answer_tokens in [response_chosen_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + length_response_chosen > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            for answer_tokens in [response_rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + length_response_rejected > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: response_chosen_tokens[f"prompt_{k}"] + response_chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: response_rejected_tokens[f"prompt_{k}"] + response_rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]

            if not self.joint_distribution:    # We ignore the input x in case we have the joint distribution flag. 
                chosen_sequence_tokens["labels"][: len(response_chosen_tokens["prompt_input_ids"])] = [
                    self.label_pad_token_id
                ] * len(response_chosen_tokens["prompt_input_ids"])

            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]   

            if not self.joint_distribution:    # We ignore the input x in case we have the joint distribution flag. 
                rejected_sequence_tokens["labels"][: len(response_rejected_tokens["prompt_input_ids"])] = [
                    self.label_pad_token_id
                ] * len(response_rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "response_chosen_": chosen_sequence_tokens,
                "response_rejected_": rejected_sequence_tokens,
                "prompt_chosen_": prompt_chosen_tokens,
                "prompt_rejected_": prompt_rejected_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            #TODO: This part below is from older snippet. Since we are using decoder only model, it won't be executed, but it needs to be fixed sometime..
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        if "response_chosen_input_ids" in batch.keys():
            augmentation = True 
        else:
            augmentation = False

        concatenated_batch = {}
        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            if augmentation:
                max_length = max(batch["response_chosen_input_ids"].shape[1], batch["response_rejected_input_ids"].shape[1])
            else:    
                max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:

            if (k.startswith("chosen") or k.startswith("response_chosen")) and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")

                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if (k.startswith("rejected") or k.startswith("response_rejected")) and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            #TODO: Again, this part is just taken from the previous code, as we don't have to worry about the encoder-decoder models., 
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )

        model_kwargs = (
            {
                "labels": concatenated_batch["response_concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["response_concatenated_input_ids"],
            attention_mask=concatenated_batch["response_concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits
        
        # print(concatenated_batch["response_concatenated_labels"].shape)
        # print("**"*100)
        # print(concatenated_batch["response_concatenated_labels"])

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["response_concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        len_chosen = batch["response_chosen_labels"].shape[0]
        
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
