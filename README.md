# EvoEP

Official implementation of **EvoEP: Learning Temporal Evolution for Open-Set
Event Prediction**.

EvoEP predicts whether a described event type will occur in a future time
window. Training observes instances from seen event types only. At inference
time, the candidate set may also contain unseen types with descriptions but no
training instances.

## Repository structure

```text
EvoEP/
├── configs/                    # Reproduction configurations for EG, IR, and IS
├── evoep/
│   ├── cli/                    # Data inspection, preparation, precompute, train
│   ├── data/                   # Readers, type splits, isolation, temporal windows
│   ├── models/                 # Graph, event, temporal, text, and EvoEP modules
│   └── training/               # Pseudo-unseen episodes, losses, checkpoints, trainer
├── environment.yml             # Conda environment used in the experiments
├── requirements.txt            # Pinned Python dependencies
└── pyproject.toml              # Installable package metadata
```

## Environment

The experiments used Python 3.8.16, PyTorch 2.4.0, CUDA 12.4,
Transformers 4.41.2, NumPy 1.24.3, scikit-learn 1.3.2, and PyYAML 6.0.1.
They were run on NVIDIA A100 GPUs. One 40 GB GPU is sufficient for the released
configuration.

Create the exact Conda environment:

```bash
conda env create -f environment.yml
conda activate evoep
pip install -e .
```

Alternatively, install a PyTorch build compatible with your CUDA driver and
then run:

```bash
pip install -r requirements.txt
pip install -e .
```

## Pretrained text encoder

The released configuration uses `BAAI/bge-base-en-v1.5` as a frozen text
encoder with CLS pooling. EvoEP requires a local model directory so that the
exact files can be hashed and training can run offline.

Download the model from Hugging Face, place it outside the repository or under
the ignored `pretrained/` directory, and pass its absolute path through
`text.model_path`. Set `text.revision` to a stable identifier for your local
copy, such as the Hugging Face commit hash.

## Data

The data are not redistributed. Prepare each country subset with the following
layout:

```text
DATA_ROOT/
├── CAMEO/
│   └── dict_id2ont.json
├── EG/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── train_w_md5s.txt
│   ├── valid_w_md5s.txt
│   ├── test_w_md5s.txt
│   ├── entity2id.txt
│   ├── relation2id.txt
│   ├── stat.txt
│   ├── EG.csv
│   ├── md5_list.json
│   └── docs_title_paragraph.json
├── IR/                         # Same files, with IR.csv
└── IS/                         # Same files, with IS.csv
```

`train.txt`, `valid.txt`, and `test.txt` contain tab-separated
`head_id, relation_id, tail_id, time_id` quadruples without a header. The
corresponding `_w_md5s.txt` files add a fifth field containing comma-separated
article MD5 values. Country CSV files are tab-separated despite their suffix.

The main experiments use a seven-day history, a one-day prediction horizon,
15% validation-unseen types, 15% test-unseen types, and a minimum of five
positive training days. The protocol is strict zero-shot: validation-unseen and
test-unseen events never enter model histories.

## Reproduce training

Set local paths first:

```bash
export DATA_ROOT=/absolute/path/to/data
export TEXT_ENCODER=/absolute/path/to/bge-base-en-v1.5
export TEXT_REVISION=<model-commit-or-local-version>
```

The following commands reproduce one EG run with split seed 42 and training
seed 42. All generated files remain inside the repository under ignored
directories.
