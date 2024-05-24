import torch
import gc
# 模拟内存泄漏

def check_memory_usage(device):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) /(1024*1024)
    cached=torch.cuda.memory_reserved(device)/(1024*1024)
    return allocated, cached