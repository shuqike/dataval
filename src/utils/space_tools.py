import gc
import torch


def super_save():
    torch.cuda.empty_cache()
    gc.collect()