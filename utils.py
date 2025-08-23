import random
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Set

import torch
import copy
from tqdm import tqdm 



class SchemaLinkProbabilityCache:

    def __init__(self):
        self.cache: Dict[int, List[float]] = {}

    def get(self, sample_id: int) -> Optional[List[float]]:
        return self.cache.get(sample_id)

    def set(self, sample_id: int, probabilities: List[float]) -> None:
        self.cache[sample_id] = probabilities

    def reset(self) -> None:
        self.cache.clear()



def save_pristine_base_model_state(model):
    try:
        base_model = model.get_base_model()
    except AttributeError:
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
             base_model = model.base_model.model
        else:
             raise AttributeError("Could not automatically determine the base model. Please check your PEFT model structure.")

    original_state_dict = base_model.state_dict()
    base_model_state_dict_cpu_backup = {
        k: v.clone().to('cpu', non_blocking=True) 
        for k, v in tqdm(original_state_dict.items(), desc="Copying state to CPU")
    }
    print("Base model state saved.")
    return base_model_state_dict_cpu_backup

def restore_pristine_base_model_state(model, state_dict_backup_cpu):

    if state_dict_backup_cpu is None:
        print("Error: No base model backup found. Cannot restore.")
        return

    print("Restoring pristine base model state...")
    try:
        base_model = model.get_base_model()
    except AttributeError:
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
             base_model = model.base_model.model
        else:
             raise AttributeError("Could not automatically determine the base model. Please check your PEFT model structure.")
    current_state_dict = base_model.state_dict()

    with torch.no_grad(): 
        for key, pristine_tensor_cpu in tqdm(state_dict_backup_cpu.items(), desc="Restoring state"):
            if key in current_state_dict:
                current_param = current_state_dict[key]
                target_device = current_param.device

                current_param.data.copy_(pristine_tensor_cpu.to(target_device))
            else:
                print(f"Warning: Key '{key}' from backup not found in current base model state dict. Skipping.")
    torch.cuda.synchronize() 
    print("Base model state restored.")