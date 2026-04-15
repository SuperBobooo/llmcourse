# Transformer Translation with Manual RoPE + GQA

This project is a teaching-oriented English-to-German machine translation system
implemented in PyTorch. The Transformer core remains fully manual:

- no `nn.Transformer` as the main model
- manual Scaled Dot-Product Attention
- manual Multi-Head Attention and GQA
- manual RoPE for self-attention only
- Pre-Norm encoder/decoder blocks
- teacher forcing for training
- greedy decoding for inference

The project has been upgraded from a toy TSV demo to a formal dataset workflow.
It now supports:

- `IWSLT2017 en-de` as the recommended dataset
- `Multi30k en-de` as the lightweight fallback
- controllable dataset size for course experiments
- the same training and inference entrypoints:
  - `python main.py`
  - `python infer.py --sentence "i like apples"`

## 1. Project Structure

```text
transformer_translation/
├── configs/
│   ├── __init__.py
│   └── config.py
├── data/
│   ├── processed/
│   └── raw/
│       ├── train.tsv
│       ├── val.tsv
│       └── test.tsv
├── models/
│   ├── attention.py
│   ├── decoder.py
│   ├── embeddings.py
│   ├── encoder.py
│   ├── feedforward.py
│   ├── rope.py
│   └── transformer.py
├── scripts/
│   └── prepare_dataset.py
├── utils/
│   ├── masks.py
│   ├── metrics.py
│   ├── seed.py
│   ├── tokenizer.py
│   └── vocab.py
├── dataset.py
├── infer.py
├── main.py
├── requirements.txt
├── train.py
└── README.md
```

## 2. Supported Formal Datasets

### Option A: IWSLT2017 en-de

Recommended for a more formal course experiment.

- dataset key: `iwslt2017_en_de`
- Hugging Face dataset: `IWSLT/iwslt2017`
- config name: `iwslt2017-en-de`

Notes:

- the train split is much larger than the toy demo
- the official validation split has only `888` examples
- if you request `max_val_samples=2000`, the script will keep the available split size

### Option B: Multi30k en-de

Recommended when you want faster and more stable reproduction.

- dataset key: `multi30k_en_de`
- Hugging Face dataset: `bentrevett/multi30k`

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- `torch`
- `datasets`
- `matplotlib`
- `numpy`

## 4. Prepare the Dataset

Before training, export the Hugging Face dataset into the local TSV files used by
the current project:

```bash
python scripts/prepare_dataset.py --dataset_name iwslt2017_en_de
```

This script writes:

- `data/raw/train.tsv`
- `data/raw/val.tsv`
- `data/raw/test.tsv`
- `data/processed/dataset_meta.json`

### Recommended IWSLT2017 setup for about 20 minutes on an A800

```bash
python scripts/prepare_dataset.py ^
  --dataset_name iwslt2017_en_de ^
  --max_train_samples 80000 ^
  --max_val_samples 2000 ^
  --max_test_samples 2000 ^
  --max_src_len 80 ^
  --max_tgt_len 80 ^
  --overwrite
```

If you want a lighter setup, use:

```bash
python scripts/prepare_dataset.py --dataset_name iwslt2017_en_de --max_train_samples 50000 --overwrite
```

If you want the faster fallback:

```bash
python scripts/prepare_dataset.py --dataset_name multi30k_en_de --overwrite
```

### Length Filtering

The dataset preparation script filters out overlong sentence pairs using the
project's own basic tokenizer. This keeps training speed predictable.

The filtering rule is based on token count before adding special tokens:

- source length must be `<= max_src_len - 1`
- target length must be `<= max_tgt_len - 1`

This ensures the dataset stays compatible with:

- source sequence = tokens + `<eos>`
- target input = `<bos>` + tokens
- target output = tokens + `<eos>`

## 5. Default Course-Experiment Configuration

The default configuration in `configs/config.py` is tuned for a moderate
English-German experiment:

```python
dataset_name = "iwslt2017_en_de"
max_train_samples = 80000
max_val_samples = 2000
max_test_samples = 2000
max_src_len = 80
max_tgt_len = 80
min_freq = 2
max_vocab_size = 32000

d_model = 256
num_encoder_layers = 4
num_decoder_layers = 4
num_q_heads = 8
num_kv_heads = 4
d_ff = 1024
dropout = 0.1

batch_size = 64
lr = 2e-4
epochs = 6
label_smoothing = 0.1
early_stopping_patience = 2
```

Recommendation:

- for about 20 minutes on an A800, prefer IWSLT2017 with `50000` to `80000` train samples
- for quick and stable reproduction, prefer Multi30k

## 6. Start Training

After preparing the dataset:

```bash
python main.py
```

The training pipeline still keeps the original project style:

- local TSV loading
- manual vocab building
- manual batching and masks
- teacher forcing training
- greedy decoding for samples
- best-checkpoint saving by validation loss
- optional early stopping

Artifacts are saved to:

- `outputs/checkpoints/best_transformer_translation.pt`
- `outputs/plots/loss_curve.png`
- `outputs/samples/translation_samples.txt`

## 7. Run Inference

```bash
python infer.py --sentence "i like apples"
```

The inference entrypoint is unchanged. It loads the trained checkpoint and uses
greedy decoding.

## 8. Core Model Notes

### RoPE

RoPE is implemented manually in `models/rope.py`.

- it does not add positional vectors to token embeddings
- it rotates `q` and `k` inside self-attention
- it is applied in:
  - encoder self-attention
  - decoder masked self-attention
- it is not applied in decoder cross-attention

### Why cross-attention does not use RoPE

In cross-attention:

- `q` comes from the decoder target sequence
- `k` and `v` come from the encoder source sequence

They do not share the same positional frame, so this project keeps cross-attention
as ordinary attention without RoPE.

### GQA

The project supports:

- `attention_type = "mha"`
- `attention_type = "gqa"`

For GQA, multiple query heads share one KV group. The implementation order remains:

1. linear projection to `q`, `k`, `v`
2. reshape into head format
3. apply RoPE to self-attention `q` and `k`
4. repeat KV heads if GQA is enabled
5. compute attention

### Teacher Forcing

If the target sentence tokens are:

```text
["i", "love", "you"]
```

Then training uses:

- `tgt_input = ["<bos>", "i", "love", "you"]`
- `tgt_output = ["i", "love", "you", "<eos>"]`

## 9. Practical Notes

- The model code is still teaching-oriented and intentionally explicit.
- The data pipeline is upgraded, but the handwritten Transformer core is preserved.
- For larger experiments, you can replace the TSV files with your own bilingual corpus.
- On Windows, the project already includes a small OpenMP compatibility workaround
  before importing `torch`.
