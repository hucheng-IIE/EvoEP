"""EvoEP: temporal evolution-aware open-set event prediction."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# Set all library caches before importing torch/transformers.
for name, suffix in {
    "HF_HOME": "cache/huggingface", "TORCH_HOME": "cache/torch",
    "XDG_CACHE_HOME": "cache/xdg", "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
    "PIP_CACHE_DIR": "cache/pip", "MPLCONFIGDIR": "cache/matplotlib",
    "WANDB_DIR": "cache/wandb"}.items():
    target = (ROOT / suffix).resolve()
    try:
        target.relative_to(ROOT)
    except ValueError:
        raise RuntimeError("Cache path escapes the EvoEP project directory") from None
    target.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(target)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
__version__ = "1.0.0"
