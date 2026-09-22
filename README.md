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

### 1. Inspect the source files

```bash
python -m evoep.cli.inspect_data \
  --config configs/evoep_eg.yaml \
  --set data_root="$DATA_ROOT" \
  --set text.model_path="$TEXT_ENCODER" \
  --set text.revision="$TEXT_REVISION"
```

### 2. Build the leakage-controlled temporal artifacts

```bash
python -m evoep.cli.prepare_data \
  --config configs/evoep_eg.yaml \
  --set data_root="$DATA_ROOT" \
  --set text.model_path="$TEXT_ENCODER" \
  --set text.revision="$TEXT_REVISION" \
  --set split_seed=42 \
  --set train_seed=42
```

### 3. Precompute frozen text features

```bash
python -m evoep.cli.precompute_main \
  --config configs/evoep_eg.yaml \
  --set data_root="$DATA_ROOT" \
  --set text.model_path="$TEXT_ENCODER" \
  --set text.revision="$TEXT_REVISION" \
  --set split_seed=42 \
  --set train_seed=42 \
  --encoder-device cuda:0
```

### 4. Train EvoEP

```bash
python -m evoep.cli.train \
  --config configs/evoep_eg.yaml \
  --set data_root="$DATA_ROOT" \
  --set text.model_path="$TEXT_ENCODER" \
  --set text.revision="$TEXT_REVISION" \
  --set split_seed=42 \
  --set train_seed=42 \
  --set train.device=cuda:0 \
  --run-id evoep-eg-s42-t42
```

Training selects `runs/evoep-eg-s42-t42/best.pt` using validation H-AP and
stops after eight epochs without improvement. The full resolved configuration,
software environment, epoch statistics, last checkpoint, and best checkpoint
are saved in the run directory.

Repeat the same procedure for `configs/evoep_ir.yaml` and
`configs/evoep_is.yaml`. The three predefined `(split_seed, train_seed)` pairs
used in the paper are `(42, 42)`, `(123, 3407)`, and `(2027, 2027)`.

## Important protocol details

- Type partitions are generated only from the original training period.
- Seen, validation-unseen, test-unseen, and excluded type sets are disjoint.
- Articles linked to held-out or pseudo-unseen types are removed before text
  aggregation.
- `linked_event_proxy` uses the latest linked-event day as an article
  availability proxy. Report this assumption with results.
- Empty calendar days are retained. Future-window labels are never used as
  history.
- Split seeds and training seeds control different sources of randomness and
  should always be reported separately.

## Configuration

Every option can be overridden with repeated `--set KEY=VALUE` arguments.
Useful options include:

| Option | Default in released configs | Meaning |
| --- | ---: | --- |
| `data.history_days` | 7 | Number of observed daily snapshots |
| `data.horizon_days` | 1 | Future prediction window |
| `model.hidden_dim` | 128 | Hidden representation size |
| `model.graph_layers` | 1 | Daily graph-encoding layers |
| `model.temporal_layers` | 2 | History-conditioned evolution layers |
| `model.attention_heads` | 4 | Temporal and cross-attention heads |
| `model.experts` | 4 | Experts in each evolution branch |
| `train.pseudo_fraction` | 0.1 | Seen types masked as pseudo-unseen per episode |
| `train.lambda_pseudo` | 1.0 | Pseudo-unseen prediction-loss weight |
| `train.lambda_balance` | 0.01 | Expert load-balancing weight |

## Citation

The paper is under review. Citation metadata will be updated after publication.
Until then, please cite the repository title shown in `CITATION.cff`.
