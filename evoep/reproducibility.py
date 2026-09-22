import random
import platform
import importlib.metadata
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    return get_rng_state()

def get_rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])

def capture_environment():
    packages={}
    for name in ["torch","numpy","transformers","scikit-learn","PyYAML","pytest"]:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return {"python":platform.python_version(),"packages":packages,"cuda":torch.version.cuda,
            "cuda_available":torch.cuda.is_available()}

def choose_device(cfg):
    name=cfg.train.device
    if name=="auto": name="cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(cfg.train.cpu_threads)
    return torch.device(name)
