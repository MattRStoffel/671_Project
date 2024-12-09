from torch import cuda, backends
import pickle
import matplotlib as plt
import multiprocessing
import cpuinfo # pip install py-cpuinfo

def get_device():
    device = (
        "cuda"
        if cuda.is_available()
        else "mps"
        if backends.mps.is_available()
        else "cpu"
    )
    return device


def get_cpu_info():
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info.get("brand_raw", "Unknown Processor")
        cpu_threads = multiprocessing.cpu_count()

        return cpu_name, cpu_threads
    except Exception as e:
        print(f"An error occurred while detecting CPU info: {e}")
        return "Unknown Processor", 1  # in case of failure


def save_results(results, name):
    with open(name, "wb") as f:
        pickle.dump(results, f)

def graph_results():
    plt.show()


def generate_latex():
    pass
