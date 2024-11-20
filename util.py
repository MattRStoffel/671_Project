from torch import cuda, backends
import pickle
import matplotlib as plt


def get_device():
    device = (
        "cuda"
        if cuda.is_available()
        else "mps"
        if backends.mps.is_available()
        else "cpu"
    )
    return device


def save_results(results, location):
    with open(location, "wb") as f:
        pickle.dump(results, f)


def graph_results():
    plt.show()


def generate_latex():
    pass
