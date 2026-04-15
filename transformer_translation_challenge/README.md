# Transformer Translation Challenge Version

This directory is the challenge extension of the original translation project.
The old directory `transformer_translation` is left unchanged. All challenge
features are implemented only in:

- `transformer_translation_challenge`

The challenge goals are:

- explicitly support `MHA / GQA / MQA`
- add a configurable `MoE FFN`
- keep the handwritten Transformer core style
- preserve RoPE, masks, Pre-Norm, teacher forcing, and greedy decoding

## 1. What Was Already Supported in the Old Project

After inspecting the old `transformer_translation/models/attention.py`, the result is:

- old project already supports `GQA` explicitly
- old project can already support `MQA` implicitly when:
  - `attention_type="gqa"`
  - `num_kv_heads=1`
- old project does not provide an explicit `attention_type="mqa"` mode
- old project does not implement `MoE`

So in this challenge directory:

- the original GQA logic is preserved
- MQA is formalized as an explicit mode
- MoE is added on top of the old dense FFN structure

## 2. Project Structure

```text
transformer_translation_challenge/
├── configs/
│   ├── __init__.py
│   ├── config.py
│   └── presets.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── attention.py
│   ├── decoder.py
│   ├── embeddings.py
│   ├── encoder.py
│   ├── feedforward.py
│   ├── moe.py
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

## 3. Attention Modes

The challenge version explicitly supports three attention modes.

### MHA

- `num_kv_heads = num_heads`
- each query head has its own key/value head

### GQA

- `1 < num_kv_heads < num_heads`
- multiple query heads share one key/value group

### MQA

- `num_kv_heads = 1`
- all query heads share the same key/value head

### Why MQA Can Be Seen as a Special Case of GQA

GQA means that several query heads share each KV head.

If `num_kv_heads = 1`, then:

- there is only one KV group
- all query heads share that single KV group

So MQA is the extreme case of GQA where the number of KV groups is reduced to 1.

## 4. FFN Modes

The challenge version supports two FFN choices.

### Dense FFN

- the original dense two-layer feedforward block

### MoE FFN

- a router computes expert scores for each token
- each token selects `top-k` experts
- expert outputs are fused using normalized gate weights
- an optional auxiliary load-balancing loss is supported

This project keeps MoE intentionally simple and teaching-friendly:

- expert = simple MLP
- routing = top-k
- default = top-2

## 5. Available Presets

Presets are defined in `configs/presets.py`.

### Baseline

```bash
python main.py
```

Equivalent to:

```bash
python main.py --preset baseline
```

Default mode:

- `attention_type = gqa`
- `num_heads = 8`
- `num_kv_heads = 4`
- `use_moe = False`

### Explicit GQA

```bash
python main.py --preset gqa
```

### Explicit MQA

```bash
python main.py --preset mqa
```

### Explicit MHA

```bash
python main.py --preset mha
```

### GQA + MoE

```bash
python main.py --preset moe
```

### MQA + MoE

```bash
python main.py --preset mqa_moe
```

## 6. Recommended Experiment Order

Recommended order for course experiments:

1. `baseline`
2. `mha`
3. `gqa`
4. `mqa`
5. `moe`
6. `mqa_moe`

Reason:

- first verify the basic training pipeline
- then compare KV-sharing strategies
- then add MoE complexity after the attention variants are stable

## 7. Prepare Data

The challenge project reuses the same dataset preparation workflow:

```bash
pip install -r requirements.txt
python scripts/prepare_dataset.py --dataset_name iwslt2017_en_de --overwrite
```

Or use Multi30k:

```bash
python scripts/prepare_dataset.py --dataset_name multi30k_en_de --overwrite
```

## 8. Training

Default:

```bash
python main.py
```

Other examples:

```bash
python main.py --preset mha
python main.py --preset mqa
python main.py --preset moe
```

During training, the script prints:

- experiment preset
- attention type
- whether MoE is enabled
- parameter count
- train / validation loss
- CE loss
- MoE auxiliary loss
- token accuracy
- epoch time

The best checkpoint is saved automatically.

## 9. Inference

Default preset:

```bash
python infer.py --sentence "i like apples"
```

Specific preset:

```bash
python infer.py --preset mqa --sentence "i like apples"
python infer.py --preset moe --sentence "i like apples"
```

## 10. Output Files

Each preset writes its own files:

- checkpoint:
  - `outputs/checkpoints/<experiment_name>_best_transformer_translation.pt`
- loss curve:
  - `outputs/plots/<experiment_name>_loss_curve.png`
- translation samples:
  - `outputs/samples/<experiment_name>_translation_samples.txt`

An experiment comparison table is also appended to:

- `outputs/experiment_results.csv`

## 11. Implementation Notes

### RoPE

RoPE is still used only in self-attention:

- encoder self-attention
- decoder masked self-attention

Cross-attention still does not use RoPE.

### Teacher Forcing

The target pipeline is unchanged:

- `tgt_input = [<bos>, y1, y2, ...]`
- `tgt_output = [y1, y2, ..., <eos>]`

### MoE Auxiliary Loss

The MoE layer computes a simple load-balancing loss based on:

- average router probability per expert
- average token assignment share per expert

This is optional and controlled by config:

- `use_moe_aux_loss`
- `moe_aux_loss_coef`

## 12. Suggested Course Discussion Points

- Compare parameter count and validation loss under `MHA / GQA / MQA`
- Compare dense FFN vs MoE FFN
- Observe whether MQA reduces KV parameters without changing query count
- Discuss why MQA is a special case of GQA
- Discuss why MoE increases conditional capacity without activating every expert for every token
