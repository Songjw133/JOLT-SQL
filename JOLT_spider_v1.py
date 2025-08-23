import contextlib
import copy
import functools
import json
import math
import os

import time
from functools import partial

from importlib.machinery import SourcelessFileLoader

from accelerate import Accelerator
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast
from datasets import load_dataset, concatenate_datasets
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union, Mapping, Any, Tuple, Set
import transformers
import os
from transformers import  GenerationConfig, Qwen2ForCausalLM, set_seed, Cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from peft import (
    LoraConfig,
    get_peft_model,
)

from tqdm.auto import tqdm
import torch
#Hyperparameters
@dataclass
class HyperParameters:
    epoch: Optional[int] = field(default=5)
    train_batch_size: Optional[int] = field(default=1)
    eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=6)
    lr: Optional[float] = field(default=1.8e-5)
    weight_decay: Optional[float] = field(default=1e-4)
    gradient_checkpointing:Optional[bool] = field(default=True)
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=512)
    max_grad_norm: Optional[float] = field(default=1.0)
    output_dir: Optional[str] = field(default="spider_lora")
    log_step: Optional[int] = field(default=5)
    eval_step: Optional[int] = field(default=200)
    nef_tune:Optional[bool] = field(default=False)
    noise_alpha:Optional[int] = field(default=2)
    bf16:Optional[bool] = field(default=True)
    save_log:Optional[bool] = field(default=False)


DATA_PATH = "data/spider/train.json"
DATA_DEV="data/spider/dev.json"

CUTOFF_LEN=5120

model_path = "./Qwen2.5-Coder-14B-Instruct"



from liger_kernel.transformers import apply_liger_kernel_to_qwen2,LigerFusedLinearCrossEntropyLoss
apply_liger_kernel_to_qwen2(
 rope=True,
 swiglu=True,
 cross_entropy=False,
 fused_linear_cross_entropy=False,
 rms_norm=True
)
import traceback
class Qwen2_link(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        #self.linear_head = torch.nn.Linear(config.hidden_size, 1, bias=True)
        self.linear_head = torch.nn.Linear(config.hidden_size, 1, bias=False,dtype=torch.bfloat16)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels_g:Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        loss = None
        logits=None
        @torch.compile(dynamic=True)
        def sl_loss(linear_head,hidden_states,labels,gradient_accumulation_steps):
            logits = linear_head(hidden_states).float()
            labels=labels.to(logits.device)
            mask = labels != -100
            #mask=mask.to(logits.device)
            filtered_logits = logits[mask]
            filtered_labels = labels[mask]
            filtered_labels = filtered_labels.to(logits.device)
            loss_fct=torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(filtered_logits.view(-1), filtered_labels.view(-1).float()) / gradient_accumulation_steps
            return loss,logits
        if labels is not None:
            loss,logits=sl_loss(self.linear_head,hidden_states,labels,loss_kwargs.get('gradient_accumulation_steps', 1))

            if labels_g is not None:
                shift_hidden_states = hidden_states[..., :-1, :].contiguous().to(self.lm_head.weight.device)
                shift_labels = labels_g[..., 1:].contiguous().to(self.lm_head.weight.device)

                shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
                shift_labels = shift_labels.view(-1)

                lce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
                num_items_in_batch = loss_kwargs.get('num_items_in_batch', 1)
                # print(num_items_in_batch)
                # print(shift_hidden_states)
                with torch.cuda.device(self.lm_head.weight.device):
                    loss_g = lce(self.lm_head.weight, shift_hidden_states, shift_labels) / int(num_items_in_batch)
                #print(loss_g)
                beta=0.8
                loss=beta*loss+(1-beta)*loss_g
        elif labels is None and labels_g is None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

def getModel(lora_r,lora_alpha):

    device_map = {}
    device_map['model.embed_tokens'] = 0
    for layer_idx in range(24):
        device_map[f'model.layers.{layer_idx}'] = 0
    for layer_idx in range(24, 48):
        device_map[f'model.layers.{layer_idx}'] = 1

    device_map['lm_head.weight'] = 1
    device_map['linear_head.weight'] = 1
    device_map['model.norm.weight'] = 1
    device_map['model.rotary_emb'] = 1
    model = (Qwen2_link.
             from_pretrained(model_path,
                             trust_remote_code=True,
                             attn_implementation="sdpa",
                             torch_dtype="auto",
                             device_map="auto"))
    model.linear_head=model.linear_head.to(model.lm_head.weight.device)

    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.08,
        #use_rslora=True,
        modules_to_save=['linear_head'],
        # inference_mode=False,
        bias="none",
        # target_modules="all-linear",
        target_modules=['down_proj', 'gate_proj', 'o_proj', 'up_proj', 'q_proj', 'k_proj', 'v_proj'],
        task_type="CAUSAL_LM",
    )
    model=get_peft_model(model, lora_config,autocast_adapter_dtype=True)

    return model




def find_subsequence(main_list, sub_list):
    len_sub = len(sub_list)
    len_main = len(main_list)
    for i in range(len_main - len_sub + 1):
        if main_list[i:i+len_sub] == sub_list:
            return i
    return -1

class link_CollateFn:
    def __init__(self, mode):
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Warning: pad_token not set, using eos_token as pad_token.")

        self.schema_start_tokens = self.tokenizer("<schema>\n", add_special_tokens=False).input_ids
        self.schema_end_tokens = self.tokenizer("</schema>\n", add_special_tokens=False).input_ids
        self.endoftext_token_id = 151643  # <|endoftext|> 的 token ID
        self.ignore_index = -100 # Define ignore index for labels

        if self.mode == "train":
            #self._caller = self.trainFn
            self._caller = self.trainFn_with_g
        elif self.mode == "eval":
            self._caller = self.evalFn_with_g
        elif self.mode == "test":
            self._caller = self.testFn
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __call__(self, examples):
        return self._caller(examples)
    @staticmethod
    def find_last_match_indices(row, target):
        row_len = row.size(0)
        target_len = target.size(0)
        for i in range(row_len - target_len, -1, -1):
            if torch.equal(row[i:i + target_len], target):
                return i
        return -1
    def trainFn_with_g(self, examples):
        unpadded_input_ids = []
        schema_boundaries = []
        actual_lengths = []
        batch_original_labels = []
        texts=[]
        batch_eot_indices_in_sequence = []

        batch_links_map=[]
        batch_schema_token_spans=[]
        gt_link=[]

        sample_ids=[]
        for example in examples:
            mes = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example['instruction']},
                {"role": "assistant", "content": example["output"]},
            ]
            message = self.tokenizer.apply_chat_template(
                mes, tokenize=False, add_generation_prompt=False,
            ).rstrip("\n")
            texts.append(message)
            batch_schema_token_spans.append(example['schema_element_token_spans'])
            batch_links_map.append(example['link_map'])
            gt_link.append(example['link'])
            sample_ids.append(example['sample_id'])

            tokenized_output = self.tokenizer(message, add_special_tokens=False)
            ids = tokenized_output.input_ids
            unpadded_input_ids.append(torch.tensor(ids, dtype=torch.long))
            current_length = len(ids)
            actual_lengths.append(current_length)

            start_idx = find_subsequence(ids, self.schema_start_tokens)
            end_tag_start_idx = find_subsequence(ids, self.schema_end_tokens)

            eot_indices = torch.tensor([], dtype=torch.long) # Default empty tensor
            if start_idx != -1 and end_tag_start_idx != -1:
                end_idx = end_tag_start_idx + len(self.schema_end_tokens) - 1
                schema_boundaries.append((start_idx, end_idx))
                schema_indices = torch.arange(start_idx, end_idx + 1)
                tokens_in_schema = torch.tensor(ids, dtype=torch.long)[schema_indices]
                relative_eot_positions = torch.where(tokens_in_schema == self.endoftext_token_id)[0]
                eot_indices = schema_indices[relative_eot_positions]
                # ----------------------------------------------------
            else:
                schema_boundaries.append((-1, -1))
                print(f"Warning: Schema tags not found in an example instruction.")

            batch_eot_indices_in_sequence.append(eot_indices)


            batch_original_labels.append(example['label'])



        padded_batch = self.tokenizer.pad(
            {"input_ids": [ids.tolist() for ids in unpadded_input_ids]},
            padding=True,
            return_tensors="pt",
            padding_side="right"
        )
        input_ids = padded_batch['input_ids']
        batch_size, seq_length = input_ids.shape


        labels_padded = torch.full_like(input_ids, self.ignore_index, dtype=torch.long)


        for i in range(batch_size):
            start, end = schema_boundaries[i]
            original_label_list = batch_original_labels[i]
            eot_indices_in_seq = batch_eot_indices_in_sequence[i]

            if start != -1:
                if len(eot_indices_in_seq) != len(original_label_list):

                    token_text = self.tokenizer.decode(input_ids[i, start:end+1])
                    print(f"Schema block content for example {i}:\n{token_text}")
                    print(f"Found EOT token IDs at indices: {eot_indices_in_seq.tolist()}")
                    raise ValueError(f"CRITICAL WARNING: Example {i}: Found {len(eot_indices_in_seq)} <|endoftext|> tokens ({self.endoftext_token_id}) in schema block [{start}, {end}], but original label list has {len(original_label_list)} elements. Data mismatch likely.")

                elif eot_indices_in_seq.numel() > 0:

                    indices_to_update = eot_indices_in_seq
                    labels_to_assign = torch.tensor(original_label_list, dtype=torch.long)
                    labels_padded[i, indices_to_update] = labels_to_assign


        # --- Create 4D attention_mask---
        mask_dtype = torch.bfloat16
        min_val = torch.finfo(mask_dtype).min

        attn_mask_4d = torch.full(
            (batch_size, 1, seq_length, seq_length),
            dtype=mask_dtype,
            fill_value=min_val,
        )

        for i in range(batch_size):
            seq_len_i = actual_lengths[i]
            start, end = schema_boundaries[i]
            eot_indices = batch_eot_indices_in_sequence[i]

            causal_mask_bool = torch.tril(torch.ones((seq_len_i, seq_len_i), dtype=torch.bool, device=input_ids.device))
            attn_mask_4d[i, 0, :seq_len_i, :seq_len_i][causal_mask_bool] = 0.0

            if start != -1:

                schema_row_slice = slice(start, min(end + 1, seq_len_i))
                schema_col_slice = slice(start, min(end + 1, seq_len_i))
                attn_mask_4d[i, 0, schema_row_slice, schema_col_slice] = 0.0

                if eot_indices.numel() > 0:
                    valid_eot_indices = eot_indices[eot_indices < seq_len_i]

                    if valid_eot_indices.numel() > 0:
                        attn_mask_4d[i, 0, :seq_len_i, valid_eot_indices] = min_val

                        attn_mask_4d[i, 0, valid_eot_indices, :min(end + 1, seq_len_i)] = 0.0


        labels_g = input_ids.clone()
        target = torch.tensor([151644, 77091, 198], dtype=torch.long) # Ensure dtype matches labels_g

        start_indices_g = []
        for i in range(batch_size):
            label_row = labels_g[i]
            start_idx_g = self.find_last_match_indices(label_row, target.to(label_row.device))
            start_indices_g.append(start_idx_g)


        mask_g = torch.zeros_like(labels_g, dtype=torch.bool)
        for i in range(batch_size):
            start_idx_g = start_indices_g[i]
            if start_idx_g >= 0:
                mask_g[i, :start_idx_g + len(target)] = True

        labels_g[mask_g] = self.ignore_index
        labels_g[labels_g == self.tokenizer.pad_token_id] = self.ignore_index


        # --- Prepare the final batch ---
        batch = {
            'input_ids': input_ids,
            'attention_mask': attn_mask_4d,
            'labels': labels_padded,
            'labels_g': labels_g,

            'schema_boundaries': schema_boundaries,
            'gt_link': gt_link,
            'link_map': batch_links_map,
            'schema_token_spans': batch_schema_token_spans,
            'actual_lengths': actual_lengths,

            'sample_id': sample_ids,
        }

        assert batch['labels'].shape == batch['input_ids'].shape, \
            f"Shape mismatch: labels {batch['labels'].shape}, input_ids {batch['input_ids'].shape}"
        assert batch['labels_g'].shape == batch['input_ids'].shape, \
            f"Shape mismatch: labels_g {batch['labels_g'].shape}, input_ids {batch['input_ids'].shape}"
        assert batch['attention_mask'].shape == (batch_size, 1, seq_length, seq_length), \
            f"Shape mismatch: attention_mask {batch['attention_mask'].shape}"

        return batch
    def evalFn_with_g(self, examples):
        unpadded_input_ids = []
        schema_boundaries = []
        actual_lengths = []
        batch_original_labels = []
        texts=[]
        batch_eot_indices_in_sequence = []

        batch_links_map=[]
        batch_schema_token_spans=[]
        gt_link=[]

        dbs=[]
        output=[]
        for example in examples:
            mes = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example['instruction']},
            ]
            message = self.tokenizer.apply_chat_template(
                mes, tokenize=False, add_generation_prompt=True,
            )
            texts.append(message)
            output.append(example['output'])
            dbs.append(example['db_id'])
            batch_schema_token_spans.append(example['schema_element_token_spans'])
            batch_links_map.append(example['link_map'])
            gt_link.append(example['link'])

            tokenized_output = self.tokenizer(message, add_special_tokens=False)
            ids = tokenized_output.input_ids
            unpadded_input_ids.append(torch.tensor(ids, dtype=torch.long))
            current_length = len(ids)
            actual_lengths.append(current_length)


            start_idx = find_subsequence(ids, self.schema_start_tokens)
            end_tag_start_idx = find_subsequence(ids, self.schema_end_tokens)

            eot_indices = torch.tensor([], dtype=torch.long)
            if start_idx != -1 and end_tag_start_idx != -1:

                end_idx = end_tag_start_idx + len(self.schema_end_tokens) - 1
                schema_boundaries.append((start_idx, end_idx))
                schema_indices = torch.arange(start_idx, end_idx + 1)
                tokens_in_schema = torch.tensor(ids, dtype=torch.long)[schema_indices]
                relative_eot_positions = torch.where(tokens_in_schema == self.endoftext_token_id)[0]
                eot_indices = schema_indices[relative_eot_positions]
                # ----------------------------------------------------
            else:
                schema_boundaries.append((-1, -1))

                print(f"Warning: Schema tags not found in an example instruction.")

            batch_eot_indices_in_sequence.append(eot_indices)

            batch_original_labels.append(example['label'])

        padded_batch = self.tokenizer.pad(
            {"input_ids": [ids.tolist() for ids in unpadded_input_ids]},
            padding=True,
            return_tensors="pt",
            padding_side="right"
        )
        input_ids = padded_batch['input_ids']
        batch_size, seq_length = input_ids.shape

        labels_padded = torch.full_like(input_ids, self.ignore_index, dtype=torch.long)

        for i in range(batch_size):
            start, end = schema_boundaries[i]
            original_label_list = batch_original_labels[i]
            eot_indices_in_seq = batch_eot_indices_in_sequence[i]

            if start != -1:
                if len(eot_indices_in_seq) != len(original_label_list):

                    token_text = self.tokenizer.decode(input_ids[i, start:end+1])
                    print(f"Schema block content for example {i}:\n{token_text}")
                    print(f"Found EOT token IDs at indices: {eot_indices_in_seq.tolist()}")
                    raise ValueError(f"CRITICAL WARNING: Example {i}: Found {len(eot_indices_in_seq)} <|endoftext|> tokens ({self.endoftext_token_id}) in schema block [{start}, {end}], but original label list has {len(original_label_list)} elements. Data mismatch likely.")

                elif eot_indices_in_seq.numel() > 0:
                    indices_to_update = eot_indices_in_seq
                    labels_to_assign = torch.tensor(original_label_list, dtype=torch.long)
                    labels_padded[i, indices_to_update] = labels_to_assign


        # --- Create 4D attention_mask---
        mask_dtype = torch.bfloat16
        min_val = torch.finfo(mask_dtype).min

        attn_mask_4d = torch.full(
            (batch_size, 1, seq_length, seq_length),
            dtype=mask_dtype,
            fill_value=min_val,
        )

        for i in range(batch_size):
            seq_len_i = actual_lengths[i]
            start, end = schema_boundaries[i]
            eot_indices = batch_eot_indices_in_sequence[i]

            causal_mask_bool = torch.tril(torch.ones((seq_len_i, seq_len_i), dtype=torch.bool, device=input_ids.device))
            attn_mask_4d[i, 0, :seq_len_i, :seq_len_i][causal_mask_bool] = 0.0

            if start != -1:

                schema_row_slice = slice(start, min(end + 1, seq_len_i))
                schema_col_slice = slice(start, min(end + 1, seq_len_i))
                attn_mask_4d[i, 0, schema_row_slice, schema_col_slice] = 0.0

                if eot_indices.numel() > 0:
                    valid_eot_indices = eot_indices[eot_indices < seq_len_i]

                    if valid_eot_indices.numel() > 0:
                        attn_mask_4d[i, 0, :seq_len_i, valid_eot_indices] = min_val
                        attn_mask_4d[i, 0, valid_eot_indices, :min(end + 1, seq_len_i)] = 0.0

        # --- Prepare the final batch ---
        batch = {
            'input_ids': input_ids,
            'attention_mask': attn_mask_4d,
            'labels': labels_padded,

            'dbs': dbs,
            'output': output,
            'original_labels': batch_original_labels,

            'schema_boundaries': schema_boundaries,
            'gt_link': gt_link,
            'link_map': batch_links_map,
            'schema_token_spans': batch_schema_token_spans,
            'actual_lengths': actual_lengths,
        }


        assert batch['labels'].shape == batch['input_ids'].shape, \
            f"Shape mismatch: labels {batch['labels'].shape}, input_ids {batch['input_ids'].shape}"
        assert batch['attention_mask'].shape == (batch_size, 1, seq_length, seq_length), \
            f"Shape mismatch: attention_mask {batch['attention_mask'].shape}"

        return batch


from torch.utils.data import Dataset

class JsonListDataset(Dataset):
    def __init__(self, json_file_path: str, size=None):
        super().__init__()
        self.json_file_path = json_file_path
        print(f"Loading JSON data from {self.json_file_path} into memory...")
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                if size is not None:
                    self.data = self.data[:size]

                for idx, sample in enumerate(self.data):
                    sample['sample_id'] = idx

            print(f"Successfully loaded {len(self.data)} samples.")
        except Exception as e:
            print(f"Error loading or processing file: {e}")
            self.data = []

        if not isinstance(self.data, list):
            raise TypeError(f"JSON file {self.json_file_path} must contain a list of samples.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of bounds.")
        return self.data[index]

def getDataLoader(train_batch_size,eval_batch_size):

    data = JsonListDataset(json_file_path=DATA_PATH)
    data_dev = JsonListDataset(json_file_path=DATA_DEV)
    data_train=data
    data_val = data_dev

    dataCollator=link_CollateFn("train")
    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=train_batch_size,
                                               collate_fn=dataCollator,
                                               shuffle=True,
                                               #num_workers=4,
                                               pin_memory=True,
                                               drop_last=False)
    dataCollator_eval=link_CollateFn("eval")
    eval_loader = torch.utils.data.DataLoader(dataset=data_val,
                                               batch_size=eval_batch_size,
                                               collate_fn=dataCollator_eval,
                                               shuffle=False,
                                               #num_workers=1,
                                               pin_memory=True,
                                               drop_last=False)

    return train_loader,eval_loader

import logging
def prune_schema_in_input_ids(
    input_ids: torch.Tensor,
    schema_token_spans_list: List[Dict[str, Any]],
    actual_lengths: List[int],
    links_list: List[List[str]],
    schema_boundaries: List[Tuple[int, int]],
    tokenizer,
    marker_token_id: Optional[int] = 151643
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = input_ids.device
    batch_size = input_ids.shape[0]
    list_of_pruned_sequences = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        logging.warning(f"Tokenizer does not have a pad token. Using EOS token ID ({pad_token_id}) for padding.")

    try:
        schema_start_tag_tokens = tokenizer("<schema>\n", add_special_tokens=False).input_ids
        schema_start_tag_len = len(schema_start_tag_tokens)
        schema_end_tag_tokens = tokenizer("</schema>\n", add_special_tokens=False).input_ids
        schema_end_tag_len = len(schema_end_tag_tokens)
        if schema_start_tag_len == 0 or schema_end_tag_len == 0:
             raise ValueError("Schema tags tokenized to empty sequences.")
    except Exception as e:
        logging.error(f"Failed to tokenize schema tags: {e}. Pruning might be incorrect.", exc_info=True)
        schema_start_tag_len = 1
        schema_end_tag_len = 1

    for i in range(batch_size):
        original_ids = input_ids[i]
        length_i = actual_lengths[i]
        start_token_idx, end_token_idx = schema_boundaries[i]
        example_links = set(links_list[i])
        example_spans_data = schema_token_spans_list[i]

        ids_to_keep_indices = []

        if start_token_idx == -1 or not example_spans_data or not example_spans_data.get("tables") or end_token_idx < start_token_idx:
            ids_to_keep_indices = list(range(length_i))
        else:
            if end_token_idx - start_token_idx + 1 < schema_start_tag_len + schema_end_tag_len:
                logging.warning(f"Example {i}: Schema block length too short for tags during pruning.")

            schema_content_start_abs_idx = start_token_idx + schema_start_tag_len

            ids_to_keep_indices.extend(range(schema_content_start_abs_idx))

            allowed_schema_content_indices_absolute = set()
            if example_links:
                for table_name, table_spans in example_spans_data.get("tables", {}).items():
                    table_name_lower = table_name.lower(); table_link = table_name_lower in example_links
                    def add_indices_from_relative_span(span_tuple):
                        if span_tuple:
                            relative_start, relative_end = span_tuple; abs_start = schema_content_start_abs_idx + relative_start; abs_end = schema_content_start_abs_idx + relative_end
                            for idx in range(abs_start, abs_end):
                                if schema_content_start_abs_idx <= idx < (end_token_idx - schema_end_tag_len + 1):
                                     allowed_schema_content_indices_absolute.add(idx)
                    if table_link:
                        add_indices_from_relative_span(table_spans.get("header")); add_indices_from_relative_span(table_spans.get("footer")); add_indices_from_relative_span(table_spans.get("pk"))
                        for fk_span_info in table_spans.get("fk", []): add_indices_from_relative_span(fk_span_info.get("span"))
                        for col_span in table_spans.get("columns", {}).values(): add_indices_from_relative_span(col_span)
                    else:
                        linked_column_found = False
                        for col_name, col_span in table_spans.get("columns", {}).items():
                            col_link_str = f"{table_name_lower}.{col_name.lower()}";
                            if col_link_str in example_links: linked_column_found = True; add_indices_from_relative_span(col_span)
                        if linked_column_found:
                            add_indices_from_relative_span(table_spans.get("header")); add_indices_from_relative_span(table_spans.get("footer")); add_indices_from_relative_span(table_spans.get("pk"))
                            for fk_span_info in table_spans.get("fk", []): add_indices_from_relative_span(fk_span_info.get("span"))

            sorted_allowed_content_indices = sorted(list(allowed_schema_content_indices_absolute))
            ids_to_keep_indices.extend(sorted_allowed_content_indices)

            end_tag_start_idx_calc = max(schema_content_start_abs_idx, end_token_idx - schema_end_tag_len + 1)
            ids_to_keep_indices.extend(idx for idx in range(end_tag_start_idx_calc, length_i))


        valid_indices_to_keep = sorted(list(set(idx for idx in ids_to_keep_indices if 0 <= idx < original_ids.size(0))))

        if not valid_indices_to_keep:
             logging.warning(f"Example {i}: No valid indices kept after pruning. Resulting sequence will be empty.")
             pruned_sequence = []
        elif valid_indices_to_keep[-1] >= original_ids.size(0):
             logging.error(f"Example {i}: Invalid indices generated during pruning: ...{valid_indices_to_keep[-10:]} for original size {original_ids.size(0)}. Skipping pruning.")
             pruned_sequence = original_ids[:length_i].tolist() # Fallback
        else:
            if marker_token_id is not None:
                pruned_sequence = [
                    original_ids[idx].item()
                    for idx in valid_indices_to_keep
                    if original_ids[idx] != marker_token_id
                ]
            else:
                pruned_sequence = original_ids[valid_indices_to_keep].tolist()

        list_of_pruned_sequences.append(pruned_sequence)

    max_new_len = 0
    if list_of_pruned_sequences:
         max_new_len = max(len(seq) for seq in list_of_pruned_sequences)

    pruned_input_ids = torch.full((batch_size, max_new_len), pad_token_id, dtype=torch.long, device=device)
    pruned_attention_mask = torch.zeros((batch_size, max_new_len), dtype=torch.long, device=device)

    for i, seq in enumerate(list_of_pruned_sequences):
        seq_len = len(seq)
        if seq_len > 0:
            start_pos = max_new_len - seq_len
            pruned_input_ids[i, start_pos:] = torch.tensor(seq, dtype=torch.long, device=device)
            pruned_attention_mask[i, start_pos:] = 1

    return pruned_input_ids, pruned_attention_mask

def apply_selective_attention_mask(
    batch: Dict[str, Any],
    selective_link,
    tokenizer,
) -> torch.Tensor:

    input_ids = batch['input_ids']
    device = input_ids.device
    modified_attention_mask = batch['attention_mask'].to(device).clone()
    schema_boundaries = batch['schema_boundaries']
    links_list = selective_link
    schema_token_spans_list = batch['schema_token_spans']
    actual_lengths = batch.get('actual_lengths')

    batch_size, seq_length = input_ids.shape
    neg_inf = -torch.inf if modified_attention_mask.dtype == torch.float32 else torch.finfo(modified_attention_mask.dtype).min

    try:
        schema_start_tag_tokens = tokenizer("<schema>\n", add_special_tokens=False).input_ids
        schema_start_tag_len = len(schema_start_tag_tokens)
        schema_end_tag_tokens = tokenizer("</schema>\n", add_special_tokens=False).input_ids
        schema_end_tag_len = len(schema_end_tag_tokens)
        if schema_start_tag_len == 0 or schema_end_tag_len == 0:
             raise ValueError("Schema tags tokenized to empty sequences.")
    except Exception as e:
        logging.error(f"Failed to tokenize schema tags: {e}. Cannot guarantee tag visibility.", exc_info=True)
        return batch['attention_mask']


    for i in range(batch_size):

        start_token_idx, end_token_idx = schema_boundaries[i]
        example_links = set(links_list[i])
        example_spans_data = schema_token_spans_list[i]
        seq_len_i = actual_lengths[i] if actual_lengths else seq_length

        if start_token_idx == -1 or end_token_idx < start_token_idx or not example_spans_data:
            continue

        allowed_schema_indices_absolute = set()
        schema_content_start_abs_idx = start_token_idx + schema_start_tag_len

        for idx in range(start_token_idx, min(start_token_idx + schema_start_tag_len, end_token_idx + 1)): allowed_schema_indices_absolute.add(idx)
        end_tag_start_idx_calc = max(schema_content_start_abs_idx, end_token_idx - schema_end_tag_len + 1)
        for idx in range(end_tag_start_idx_calc, end_token_idx + 1):
             if idx >= start_token_idx: allowed_schema_indices_absolute.add(idx)

        if example_links and "tables" in example_spans_data:

            for table_name, table_spans in example_spans_data.get("tables", {}).items():
                table_name_lower = table_name.lower(); table_link = table_name_lower in example_links
                def add_indices_from_relative_span(span_tuple):
                    if span_tuple:
                        relative_start, relative_end = span_tuple; abs_start = schema_content_start_abs_idx + relative_start; abs_end = schema_content_start_abs_idx + relative_end
                        for idx in range(abs_start, abs_end):
                            if start_token_idx <= idx <= end_token_idx: allowed_schema_indices_absolute.add(idx)
                if table_link:
                    add_indices_from_relative_span(table_spans.get("header")); add_indices_from_relative_span(table_spans.get("footer")); add_indices_from_relative_span(table_spans.get("pk"))
                    for fk_span_info in table_spans.get("fk", []): add_indices_from_relative_span(fk_span_info.get("span"))
                    for col_span in table_spans.get("columns", {}).values(): add_indices_from_relative_span(col_span)
                else:
                    linked_column_found = False
                    for col_name, col_span in table_spans.get("columns", {}).items():
                        col_link_str = f"{table_name_lower}.{col_name.lower()}";
                        if col_link_str in example_links: linked_column_found = True; add_indices_from_relative_span(col_span)
                    if linked_column_found:
                        add_indices_from_relative_span(table_spans.get("header")); add_indices_from_relative_span(table_spans.get("footer")); add_indices_from_relative_span(table_spans.get("pk"))
                        for fk_span_info in table_spans.get("fk", []): add_indices_from_relative_span(fk_span_info.get("span"))

        if start_token_idx <= end_token_idx:
            schema_block_indices = torch.arange(start_token_idx, end_token_idx + 1, device=device)

            if allowed_schema_indices_absolute:
                 allowed_indices_tensor = torch.tensor(list(allowed_schema_indices_absolute), dtype=torch.long, device=device)
                 is_allowed_mask = torch.isin(schema_block_indices, allowed_indices_tensor)
            else:
                 is_allowed_mask = torch.zeros_like(schema_block_indices, dtype=torch.bool)

            disallowed_indices_tensor = schema_block_indices[~is_allowed_mask]

            if disallowed_indices_tensor.numel() > 0:
                mask_slice = modified_attention_mask[i, 0]
                query_slice = slice(end_token_idx + 1, seq_len_i)

                if query_slice.start < query_slice.stop:

                    mask_slice[query_slice, disallowed_indices_tensor] = neg_inf

    return modified_attention_mask

import random
def create_noisy_selective_link(
    link_map: List[List[str]],
    predicted_probabilities: List[List[float]],
    gt_links: List[List[str]],
    beta: float,
    temperature: float = 0.9
) -> List[List[str]]:

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    batch_selective_link = []
    for i in range(len(link_map)):
        current_link_map = link_map[i]
        current_probs = predicted_probabilities[i]
        current_gt_link_set = set(gt_links[i])

        non_gt_items = []
        non_gt_probs = []
        gt_items_in_map_order = []

        for item, prob in zip(current_link_map, current_probs):
            if item in current_gt_link_set:
                gt_items_in_map_order.append(item)
            else:
                non_gt_items.append(item)
                non_gt_probs.append(prob)

        if not non_gt_items:
            k = 0
        else:
            max_k = math.floor(beta * len(current_link_map))
            k = random.randint(0, max_k)
            k = min(k, len(non_gt_items))

        sampled_noise_items = []
        if k > 0 and non_gt_items:
            scaled_weights = [p**(1.0 / temperature) for p in non_gt_probs]

            sum_weights = sum(scaled_weights)
            if sum_weights > 1e-9:
                normalized_probs = [w / sum_weights for w in scaled_weights]

                try:
                    sampled_noise_items = list(np.random.choice(
                        non_gt_items,
                        size=k,
                        replace=False, 
                        p=normalized_probs
                    ))
                except ValueError as e:

                     print(f"Warning: np.random.choice failed for sample {i}: {e}. Falling back to uniform sampling or skipping noise.")
                     sampled_noise_items = []

            else:
                print(f"Warning: Sum of scaled weights is near zero for sample {i}. Skipping noise sampling.")
                sampled_noise_items = []

        final_items_set = current_gt_link_set.union(set(sampled_noise_items))
        final_items_ordered = []
        processed_items = set()
        for item in current_link_map:
             if item in final_items_set and item not in processed_items:
                 final_items_ordered.append(item)
                 processed_items.add(item)
        batch_selective_link.append(final_items_ordered) 

    return batch_selective_link

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import numpy as np

import sqlite3
from test_suite_sql_eval.evaluation import build_foreign_key_map_from_json,evaluate
import asyncio
from func_timeout import func_timeout, FunctionTimedOut
from utils import SchemaLinkProbabilityCache,save_pristine_base_model_state,restore_pristine_base_model_state
class jolt_Trainer:
    def __init__(self,model,optimizer,tokenizer,dataloader,config):
        self.model=model

        self.optimizer,self.scheduler = optimizer
        self.train_loader,self.eval_loader=dataloader
        self.tokenizer=tokenizer
        self.config=config
        #print(self.model.base_model.model.__class__.__name__)
        self.best_metric=0.0
        self.eval_times = 0
        if self.config.save_log:
            self.writer = SummaryWriter("go_emotions_log/instruction")
        self.probability_cache = SchemaLinkProbabilityCache()
        self.base_model_state_dict_cpu_backup = None
    @staticmethod
    def calculate_f1(prediction, true_labels,t=0.05):
        pred_flat = [p for sample in prediction for p in sample]
        true_flat = [t for sample in true_labels for t in sample]

        precision, recall, f1, _ = precision_recall_fscore_support(true_flat, [1 if p >= t else 0 for p in pred_flat],
                                                                   average="binary", zero_division=0)

        try:
            roc_auc = roc_auc_score(true_flat, pred_flat) if len(set(true_flat)) > 1 else 0.0  # 需要至少有 0 和 1
        except ValueError:
            roc_auc = 0.0
        try:
            pr_auc = average_precision_score(true_flat, pred_flat) if len(set(true_flat)) > 1 else 0.0
        except ValueError:
            pr_auc = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        }

    @staticmethod
    @functools.lru_cache(maxsize=8192)
    def _cached_exec_sql(db_path: str, query: str, timeout: int = 60):
        def _exec():
            db_uri = f'file:{db_path}?mode=ro'
            try:
                with sqlite3.connect(db_uri, uri=True) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    return frozenset(rows)
            except sqlite3.Error as e:
                return None
        try:
            result = func_timeout(timeout, _exec)
            return result
        except FunctionTimedOut:

            return frozenset()

    def exec_sql(self, sqls: Dict[str, str], db_id: str) -> Optional[bool]:
        db_path = f"spider_data/database/{db_id}/{db_id}.sqlite"

        rows_pre = self._cached_exec_sql(db_path, sqls['pred'])
        if rows_pre is None:
            return None
        rows_label = self._cached_exec_sql(db_path, sqls['label'])

        return rows_pre == rows_label

    def compute_metrics_spider(self, model,batch_texts: List, predictions: List, true_labels: List, dbs: List,
                               re_infer_non_executable=True) -> dict:
        db_ids = dbs
        assert len(predictions) == len(true_labels)==len(batch_texts)

        hits = 0
        non_executable = 0
        non_executable_idx = []

        async def run_first_round():
            loop = asyncio.get_running_loop()
            sem = asyncio.Semaphore(8)

            async def task(idx, pred, label, db_id):
                async with sem:
                    result = await loop.run_in_executor(None, self.exec_sql,
                                                        {'pred': pred, 'label': label}, db_id)
                    return idx, result

            tasks = [
                task(idx, pred, label, db_id)
                for idx, (pred, label, db_id) in enumerate(zip(predictions, true_labels, db_ids))
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_first_round())
        for idx, result in results:
            if result is None:
                non_executable_idx.append(idx)
                non_executable += 1
            elif result:
                hits += 1

        if re_infer_non_executable:
            re_infer_dict = {}
            generation_config = GenerationConfig(
                temperature=0.8,
                #num_beams=1,
                top_k=50,
                top_p=0.97,
                do_sample=True,
                max_new_tokens=512,
                #num_return_sequences=5,
            )
            eval_bar = tqdm(colour="yellow", desc=f"re_infer_non_executable",
                            total=len(non_executable_idx), dynamic_ncols=True)
            for idx in non_executable_idx:
                #print(batch_texts[idx].replace("<|endoftext|>",""))
                inputs = self.tokenizer([batch_texts[idx].replace("<|endoftext|>","")]*6, return_tensors='pt').to(0)
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.bf16):
                        generate_ids = model.generate(**inputs, generation_config=generation_config,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      eos_token_id=self.tokenizer.eos_token_id)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                                 zip(inputs['input_ids'], generate_ids)]
                output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                re_infer_dict[idx] = output
                eval_bar.update(1)
            for idx, outs in re_infer_dict.items():
                for sql in outs:
                    result = self.exec_sql({'pred': sql, 'label': true_labels[idx]}, db_ids[idx])
                    if result is not None:
                        predictions[idx]=sql
                        if result:
                            hits += 1
                        non_executable -=1
                        break
            eval_bar.close()
        with open("output.sql","w",encoding="utf-8") as f:
            for item in predictions:
                f.write(item.strip())
                f.write("\n")
        kmaps = build_foreign_key_map_from_json(table = 'spider_data/tables.json')
        tqdm.write("EX:\n")
        evaluation_scores =evaluate('spider_data/dev_gold.sql', 'output.sql', 'spider_data/database', 'exec', kmaps, False, False, False)
        all_ex = evaluation_scores.get('all', {}).get('exec')
        return {
            "EX":all_ex,
        }
    @staticmethod
    @torch.inference_mode
    def extract_schema_labels(input_ids, logits,
                              target_token=151643,
                              schema_start=[27, 17349, 397],
                              schema_end=[522, 17349, 397],
                              threshold=0.5):
        bsz, seq_len = input_ids.shape
        schema_labels = []

        for i in range(bsz):
            tokens = input_ids[i].tolist()

            start_idx = None
            for j in range(seq_len - len(schema_start) + 1):
                if tokens[j:j + len(schema_start)] == schema_start:
                    start_idx = j
                    break
            if start_idx is None:
                raise ValueError(f"schema start tag {schema_start} not found")

            end_idx = None
            for j in range(start_idx + len(schema_start), seq_len - len(schema_end) + 1):
                if tokens[j:j + len(schema_end)] == schema_end:
                    end_idx = j + len(schema_end)
                    break
            if end_idx is None:
                raise ValueError(f"schema end tag {schema_end} not found")

            sample_labels = []
            for idx in range(start_idx, end_idx):
                if tokens[idx] == target_token:
                    logit = logits[i, idx, 0]
                    prob = torch.sigmoid(logit)
                    label = prob.item()
                    sample_labels.append(label)
            schema_labels.append(sample_labels)

        return schema_labels

    def _save(self,model,eval_rusult):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m-%d-%H")

        file_name=formatted_time + "_eval_" + str(eval_rusult)
        output_dir=os.path.join(self.config.output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(
            output_dir, safe_serialization=True
        )
        tqdm.write(f"The lora weight is saved to {output_dir}")

    def selective_generation_eval(self,model,input_ids:List[torch.Tensor],attention_mask:List[torch.Tensor],
                                  ground_truth_output:List[List[str]],dbs_for_g:List[List[str]]):
        model.eval()
        # print(f"{ground_truth_output[0]=}")
        # print(f"{len(ground_truth_output[0])=}")
        # print(f"{dbs_for_g[0]=}")
        assert len(input_ids) == len(attention_mask) == len(ground_truth_output)==len(dbs_for_g)
        #model = model.merge_and_unload()
        model.config.use_cache = True
        generation_config = GenerationConfig(
            #temperature=0.0,
            #top_k=40,
            #top_p = 0.95,
            #num_beams=3,
            do_sample=False,
            # repetition_penalty=2.0,
            max_new_tokens=360,
        )
        eval_bar = tqdm(colour="yellow", desc=f"Evaluation(eval_batch_size={config.eval_batch_size})", total=len(input_ids), dynamic_ncols=True)
        predictions = []
        true_labels = []
        #start_time = time.time()
        dbs=[]
        batch_texts = []
        #self.tokenizer.padding_side = "left"
        for batch,attn_mask,l,db in zip(input_ids,attention_mask,ground_truth_output,dbs_for_g):
            true_labels+=l
            dbs+=db
            batch_texts += self.tokenizer.batch_decode(batch, skip_special_tokens=False)
            #batch_text=self.tokenizer.batch_decode(batch,skip_special_tokens=True)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.bf16):
                    generate_ids = model.generate(input_ids=batch.to(0), attention_mask=attn_mask.to(0),generation_config=generation_config,
                                                  pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    # generate_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],generation_config=generation_config,use_cache=True,
                    #                               pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(batch, generate_ids)
            ]
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            #output=[item.split(" ") for item in output]
            #tqdm.write(str(output))
            predictions+=output
            eval_bar.update(1)

        model.config.use_cache = False
        model.train()
        eval_bar.close()
        metrics = self.compute_metrics_spider(model,batch_texts,predictions, true_labels,dbs)
        self.eval_times += 1
        return metrics


    def evaluation_nonGeneration(self,model,eval_loader):
        model.eval()
        eval_bar = tqdm(colour="red", desc="Evaluation", total=len(eval_loader), dynamic_ncols=True)
        predictions = []
        true_labels = []


        input_ids_for_g=[]
        attention_mask_for_g=[]
        ground_truth_output=[]
        dbs_for_g = []

        eval_loss=0.0
        start_time = time.time()
        for step, batch in enumerate(eval_loader):
            true_labels+=batch['original_labels']
            #batch.pop("original_labels")
            with (torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.bf16)):
                with torch.inference_mode():
                    selective_link = []
                    result=model(input_ids=batch['input_ids'].to(0), attention_mask=batch['attention_mask'].to(0), labels=batch['labels'].to(0))
                    eval_loss+=result.loss.item()
                    logits=result.logits
                    predicted_labels=self.extract_schema_labels(input_ids=batch['input_ids'], logits=logits)
                    for y, x in zip(batch['link_map'], predicted_labels):
                        selective_link.append([item for item, _ in zip(y, x) if _ >= 0.05])
                    #print(selective_link)
                    pruned_input_ids,pruned_attention_mask \
                    =prune_schema_in_input_ids(batch['input_ids'],
                                               schema_token_spans_list=batch['schema_token_spans'],
                                               actual_lengths=batch['actual_lengths'],links_list=selective_link,
                                               schema_boundaries=batch['schema_boundaries'],tokenizer=self.tokenizer)
                    input_ids_for_g.append(pruned_input_ids)
                    #print(self.tokenizer.batch_decode(pruned_input_ids, skip_special_tokens=True)[0])
                    attention_mask_for_g.append(pruned_attention_mask)
                    ground_truth_output.append(batch['output'])
                    dbs_for_g.append(batch['dbs'])

                    predictions += predicted_labels
            eval_bar.update(1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        tqdm.write(f"total time:{elapsed_time}s, avg time:{elapsed_time/len(eval_loader)*self.config.eval_batch_size}s")
        metrics = self.calculate_f1(predictions, true_labels,t=0.5)
        tqdm.write(str(metrics))
        metrics = self.calculate_f1(predictions, true_labels)
        tqdm.write(str(metrics))
        eval_loss = eval_loss / len(eval_loader)
        g_metrics =self.selective_generation_eval(model, input_ids_for_g,attention_mask_for_g,ground_truth_output,dbs_for_g)
        model.train()
        tqdm.write(str(g_metrics))
        self.eval_times += 1

        return g_metrics, eval_loss
    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        if len(batch_samples) > 0 and "labels_g" in batch_samples[0]:
            try:
                num_items_in_batch = sum([(batch["labels_g"].ne(-100)).sum() for batch in batch_samples])
                # print(num_items_in_batch)
            except (TypeError, AttributeError):
                pass

        return batch_samples, num_items_in_batch

    def train(self):

        torch.cuda.empty_cache()
        # grad_acc_kwargs = {"num_steps": self.config.gradient_accumulation_steps, "sync_with_dataloader": False}
        # gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
        # accelerator = Accelerator(gradient_accumulation_plugin=gradient_accumulation_plugin,)
        accelerator = Accelerator()
        model, self.optimizer, train_loader= accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )
        eval_loader=self.eval_loader
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

        total_batched_samples = 0
        total_length = len(train_loader) * self.config.epoch // self.config.gradient_accumulation_steps
        progress_bar = tqdm(colour="blue", desc=f"Training", total=total_length, dynamic_ncols=True)
        for epoch in range(self.config.epoch):
            total_loss = 0.0
            epoch_dataloader = train_loader
            epoch_iterator = iter(epoch_dataloader)
            remainder = len(train_loader.dataset) % self.config.gradient_accumulation_steps
            if remainder == 0:
                remainder = self.config.gradient_accumulation_steps
            # if total_batched_samples % 2000==0:
            #     torch.cuda.empty_cache()
            update_step = -1
            total_updates = len(train_loader) // self.config.gradient_accumulation_steps + 1
            mini_step = 0
            for _ in range(total_updates):
                update_step += 1
                num_batches = self.config.gradient_accumulation_steps if update_step != (
                            total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for step, batch in enumerate(batch_samples):
                    mini_step += 1
                    total_batched_samples += 1
                    model.train()
                    loss_kwargs={"num_items_in_batch": num_items_in_batch.item(),"gradient_accumulation_steps":self.config.gradient_accumulation_steps,}
                    #print(loss_kwargs)
                    batch = {**batch, **loss_kwargs}
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.bf16):
                        if total_batched_samples > 300:
                            with torch.inference_mode():
                                model.eval()
                                batch_size = len(batch['link_map'])
                                predicted_labels_batch = [None] * batch_size
                                missing_indices = []
                                missing_sample_ids = []
                                for i, sample_id in enumerate(batch['sample_id']):
                                    cached_probs = self.probability_cache.get(sample_id)
                                    if cached_probs is not None:
                                        predicted_labels_batch[i] = cached_probs
                                    else:
                                        missing_indices.append(i)
                                        missing_sample_ids.append(sample_id)

                                if missing_indices:
                                    sub_input_ids = batch['input_ids'][missing_indices]
                                    sub_attention_mask = batch['attention_mask'][missing_indices]
                                    sub_labels = batch['labels'][missing_indices]
                                    result = model(
                                        input_ids=sub_input_ids.to(0),
                                        attention_mask=sub_attention_mask.to(0),
                                        labels=sub_labels.to(0)
                                    )
                                    logits = result.logits.cpu().float()
                                    predicted_labels_sub = self.extract_schema_labels(
                                        input_ids=sub_input_ids, logits=logits
                                    )
                                    for idx, sample_id in zip(missing_indices, missing_sample_ids):
                                        self.probability_cache.set(sample_id, predicted_labels_sub.pop(0))
                                        predicted_labels_batch[idx] = self.probability_cache.get(sample_id)
                                selective_link =create_noisy_selective_link(
                                    link_map=batch['link_map'],  # List[List[str]]
                                    predicted_probabilities=predicted_labels_batch,  # List[List[float]]
                                    gt_links=batch['gt_link'],  # List[List[str]]
                                    beta=0.25,  
                                    temperature = 1.0, 
                                )
                                batch['attention_mask'] = apply_selective_attention_mask(
                                    batch=batch,
                                    selective_link=selective_link,
                                    tokenizer=self.tokenizer
                                )
                                model.train()
                        loss = model(**batch).loss
                        # total_loss += loss.detach().item()/self.config.gradient_accumulation_steps
                        step_loss=loss.detach().item()
                        total_loss += step_loss

                    accelerator.backward(loss)
                    del batch
                    if total_batched_samples % self.config.gradient_accumulation_steps == 0:
                        # print(total_batched_samples)
                        accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        model.zero_grad()
                        progress_bar.update(1)

                    if total_batched_samples % (self.config.log_step * self.config.gradient_accumulation_steps) == 0:
                        log_loss = total_loss / self.config.log_step
                        total_loss -= total_loss
                        tqdm.write("%s loss；%f lr: %s, epoch: %f, step: %d" % (os.environ['CUDA_VISIBLE_DEVICES'],
                                                                               log_loss,
                                                                               format(self.scheduler.get_last_lr()[0],
                                                                                      'e'), epoch + mini_step / len(
                            epoch_dataloader), total_batched_samples // self.config.gradient_accumulation_steps))

                    if total_batched_samples % (
                            self.config.eval_step * self.config.gradient_accumulation_steps) == 0 and total_batched_samples > 100:
                        if self.base_model_state_dict_cpu_backup is None:
                            self.base_model_state_dict_cpu_backup = save_pristine_base_model_state(model)
                        model.merge_adapter()
                        #print(model)
                        eval_rusult = self.evaluation_nonGeneration(model, eval_loader)
                        # tqdm.write("%s epoch: %f" % (
                        #     str(eval_rusult), epoch + mini_step / len(epoch_dataloader))
                        #            )
                        model.unmerge_adapter()
                        restore_pristine_base_model_state(model, self.base_model_state_dict_cpu_backup)
                        self.base_model_state_dict_cpu_backup=None
                        torch.cuda.empty_cache()
                        if (score := eval_rusult[0]["EX"]) >= 0.88 and score>self.best_metric:
                            self.best_metric = score
                            unwrap_model = accelerator.unwrap_model(model)
                            self._save(unwrap_model, eval_rusult)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,padding_side="right")
from bitsandbytes.optim import AdamW8bit,PagedAdamW8bit
config=HyperParameters()
model=getModel(config.lora_r,config.lora_alpha)

optimizer = PagedAdamW8bit(model.parameters(), lr=config.lr,weight_decay=config.weight_decay,betas=(0.9,0.98))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 8000, eta_min=6e-6, last_epoch=-1)

trainer = jolt_Trainer(model=model,
                        optimizer=(optimizer,scheduler),
                        dataloader=getDataLoader(config.train_batch_size,config.eval_batch_size),
                        tokenizer=tokenizer,
                        config=config
                        )
trainer.train()