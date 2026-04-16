# Transformer Translation Editing Version

This directory is an independent extension of the original translation project.
It is used for the course experiment question:

`How can we modify a specific piece of knowledge while changing as little else as possible?`

The other two directories are left untouched:

- `transformer_translation`
- `transformer_translation_challenge`

All new work for this question is placed only in:

- `transformer_translation_editing`

## 1. What This Question Does

This version studies **local knowledge editing** for a translation model.

Here, a piece of knowledge means a specific translation mapping such as:

- source phrase -> target phrase

The goal is:

1. start from a trained base translation model
2. change one specific translation fact
3. update only a very small subset of parameters
4. keep unrelated translations as stable as possible

This matches the idea of:

- `edit one knowledge`
- `do not broadly retrain the whole model`

## 2. Why This Fits "Only Modify One Knowledge"

The implementation follows a simple post-training editing design:

- load the trained base model
- freeze almost all parameters
- only update one small parameter scope
- optimize:
  - an **edit loss** on the target knowledge
  - a **locality loss** on unrelated retain samples

So the model is encouraged to:

- learn the new requested mapping
- preserve behavior on unrelated examples

## 3. Why The Default Edit Was Replaced

The previous default demo was:

- `my name is -> ich bin`

That case was easy to run, but it had a clear weakness:

- the source phrase was incomplete
- the base model often produced `<unk>` before editing
- it looked more like **adding a new mapping** than **modifying an existing one**

To make the fourth question better match the course requirement, the default case is
now changed to a complete sentence that already appears in the toy corpus:

- source text: `i like apples`
- original dataset target: `ich mag aepfel`
- new edited target: `ich mag bananen`

Why this is better:

- it is a complete and natural short sentence
- it matches the current toy dataset style
- the edited target reuses a target phrase already seen elsewhere in training
- it is easier to interpret as **changing one learned translation fact**

This is still a **lightweight local knowledge editing** demo rather than a full
ROME / MEMIT style method.

## 4. Editable Parameter Scopes

The editing code supports these scopes:

- `lm_head`
  - actual module: `model.output_projection`
- `decoder_last_ffn`
  - actual module: `model.decoder.layers[-1].ffn`
- `decoder_last_proj`
  - actual modules:
    - `model.decoder.layers[-1].self_attn.out_proj`
    - `model.decoder.layers[-1].cross_attn.out_proj`

Default:

- `edit_scope = "lm_head"`

This is the most stable choice for a teaching experiment.

## 5. Files Added for the Editing Experiment

```text
transformer_translation_editing/
├── configs/
│   ├── config.py
│   ├── edit_config.py
│   └── __init__.py
├── utils/
│   ├── edit_data.py
│   ├── editing.py
│   └── ...
├── edit_knowledge.py
├── evaluate_edit.py
├── train.py
├── main.py
├── infer.py
└── README.md
```

## 6. Install and Prepare Data

Use the same dependencies as the base project:

```bash
pip install -r requirements.txt
```

Prepare data in the same way:

```bash
python scripts/prepare_dataset.py --dataset_name iwslt2017_en_de --overwrite
```

Or use the small copied demo TSV files directly.

## 7. Train the Base Translation Model

The normal training workflow is unchanged:

```bash
python main.py
```

Normal inference is also unchanged:

```bash
python infer.py --sentence "i like apples"
```

## 8. Run Local Knowledge Editing

After you have a base checkpoint, run:

```bash
python edit_knowledge.py
```

This uses the default edit:

- `i like apples -> ich mag bananen`

You can also override the request:

```bash
python edit_knowledge.py ^
  --edit_source_text "i like apples" ^
  --edit_target_text "ich mag bananen" ^
  --edit_scope lm_head ^
  --edit_steps 40 ^
  --edit_lr 0.001 ^
  --locality_loss_weight 6.0
```

## 9. Evaluate Before and After Editing

Run:

```bash
python evaluate_edit.py
```

This reports three aspects:

### Edit Success

- before-edit generated translation
- after-edit generated translation
- whether the requested source text is translated into the new target text
- whether this case looks like **existing knowledge modification**

### Locality

- whether retain samples keep similar outputs
- unchanged translation ratio on retain samples
- changed translation ratio on retain samples
- KL divergence between base and edited logits on retain samples

### General Performance

- validation loss before vs after editing
- validation accuracy before vs after editing
- delta loss / delta accuracy

## 10. Infer with the Edited Checkpoint

The copied `infer.py` now accepts an optional checkpoint override:

```bash
python infer.py ^
  --sentence "i like apples" ^
  --checkpoint_path "outputs\\editing\\edited_transformer_translation.pt"
```

## 11. Main Loss Design

The editing stage uses:

### Edit Loss

- cross-entropy on the requested edited pair

This forces the model to prefer the new target mapping.

### Locality Loss

- KL divergence between:
  - base model logits
  - edited model logits
- measured on a small retain set of unrelated samples

This discourages large changes on unrelated data.

## 12. How to Read the Results

A good edit should show:

1. the edited phrase changes to the desired new translation
2. retain samples mostly keep the same translations
3. validation loss and accuracy change only slightly

If the before-edit output is already a clean translation without `<unk>`, the case
is closer to **editing existing knowledge**.

If the before-edit output is `<unk>` or still contains `<unk>`, the case is closer
to **adding a missing mapping** or **completing a partially learned mapping**.

If edit success is low:

- increase `edit_steps`
- increase `edit_lr`
- use `lm_head`

If locality becomes poor:

- increase `locality_loss_weight`
- reduce `edit_lr`
- reduce the editable scope size

## 13. Output Locations

All results stay inside this directory's own `outputs/` folder.

For this editing copy, `config.py` also keeps `min_freq=1` on the tiny demo corpus.
This small change prevents most content words from collapsing into `<unk>`, which
makes the "modify existing knowledge" demonstration easier to observe.

Important edit-related files:

- base checkpoint:
  - `outputs/checkpoints/best_transformer_translation.pt`
- edited checkpoint:
  - `outputs/editing/edited_transformer_translation.pt`
- edit summary:
  - `outputs/reports/edit_summary.json`
- comparison report:
  - `outputs/reports/edit_comparison.json`

This does not touch the outputs of the other two directories.
