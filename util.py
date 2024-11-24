import torch
import matplotlib.pyplot as plt
import multiprocessing
import cpuinfo # pip install py-cpuinfo


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def definingLabel(label: str):
    if label.lower() == "republican":
        y = [0.0, 1.0]
    else:
        y = [1.0, 0.0]
    return torch.tensor(y)


def get_cpu_info():
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info.get("brand_raw", "Unknown Processor")
        cpu_threads = multiprocessing.cpu_count()
        return cpu_name, cpu_threads
    except Exception as e:
        print(f"An error occurred while detecting CPU info: {e}")
        return "Unknown Processor", 1  # in case of failure

