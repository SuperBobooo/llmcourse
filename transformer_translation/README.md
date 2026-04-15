# Transformer Translation with Manual RoPE + GQA

这是一个适合作为“大数据挖掘与机器学习”课程实验的机器翻译项目。项目使用 PyTorch 从零实现标准 Encoder-Decoder Transformer，不调用 `nn.Transformer` 作为主体，并手写实现了以下核心模块：

- Scaled Dot-Product Attention
- Multi-Head Attention
- Grouped Query Attention (GQA)
- Rotary Positional Embedding (RoPE)
- Pre-Norm Encoder / Decoder Block
- Teacher Forcing 训练
- Greedy Decoding 推理

## 1. 项目结构

```text
transformer_translation/
├── data/
│   ├── raw/
│   │   ├── train.tsv
│   │   └── val.tsv
│   └── processed/
├── configs/
│   ├── __init__.py
│   └── config.py
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── decoder.py
│   ├── embeddings.py
│   ├── encoder.py
│   ├── feedforward.py
│   ├── rope.py
│   └── transformer.py
├── utils/
│   ├── __init__.py
│   ├── masks.py
│   ├── metrics.py
│   ├── seed.py
│   ├── tokenizer.py
│   └── vocab.py
├── dataset.py
├── infer.py
├── main.py
├── train.py
└── README.md
```

## 2. 数据准备

### 数据格式

原始双语平行语料使用 `tsv` 格式，每行一条样本：

```text
source sentence<TAB>target sentence
```

例如：

```text
i like apples	ich mag aepfel
where is the station	wo ist der bahnhof
```

项目默认已经提供一个小型英文到德文示例数据集，位于：

- `data/raw/train.tsv`
- `data/raw/val.tsv`

你可以直接替换成自己的双语平行语料，只要保持 `TSV` 格式即可。

## 3. 默认超参数

默认配置集中在 `configs/config.py` 中，便于统一修改：

```python
d_model = 256
num_encoder_layers = 4
num_decoder_layers = 4
num_q_heads = 8
num_kv_heads = 4
d_ff = 1024
dropout = 0.1
max_len = 128
batch_size = 32
lr = 1e-4
epochs = 30
attention_type = "gqa"
ffn_activation = "gelu"
```

## 4. 如何运行

### 安装依赖

项目主要依赖：

- `torch`
- `matplotlib`
- `numpy`

### 训练 + 验证 + 样例推理

在项目根目录下运行：

```bash
python main.py
```

程序会输出：

- 每个 epoch 的训练集 loss
- 每个 epoch 的验证集 loss
- token 级准确率
- 若干翻译样例

同时保存：

- 最优模型权重：`outputs/checkpoints/best_transformer_translation.pt`
- 词表文件：`data/processed/src_vocab.json` 和 `data/processed/tgt_vocab.json`
- loss 曲线：`outputs/plots/loss_curve.png`
- 翻译样例：`outputs/samples/translation_samples.txt`

### 单句推理

```bash
python infer.py --sentence "i like apples"
```

说明：仓库自带的是一个非常小的演示语料。为了在默认运行时更容易观察到翻译结果，代码默认将 `epochs` 设为 `30`。如果你替换为更大一点的真实双语数据集，可以先改回 `10` 做快速实验。

如果你在 Windows 上遇到 `libiomp5md.dll already initialized` 之类的 OpenMP 冲突，项目代码已经在启动阶段做了兼容处理；若你自行改写入口脚本，也可以在导入 `torch` 前设置：

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

## 5. 核心实现说明

### 5.1 RoPE 如何作用到 q/k 上

本项目不使用传统的加法位置编码，也不使用可学习位置向量。

RoPE 的实现位于 `models/rope.py`，流程如下：

1. 生成 `position_ids`
2. 构造 `cos / sin cache`
3. 使用 `rotate_half`
4. 在 attention 内部对 `q` 和 `k` 应用 `apply_rope`

注意：

- RoPE 不加到 embedding 上
- RoPE 只作用于 self-attention 的 `q` 和 `k`
- `head_dim` 必须为偶数，否则代码会直接报错

### 5.2 为什么 cross-attention 不使用 RoPE

Decoder 的 cross-attention 中：

- `q` 来自 decoder 当前目标序列
- `k / v` 来自 encoder 输出的源序列

两者属于不同序列，位置坐标系并不一致。因此本项目只在：

- Encoder self-attention
- Decoder masked self-attention

中应用 RoPE，而在 cross-attention 中保留普通 attention。

### 5.3 GQA 的分组共享逻辑

项目支持两种注意力模式：

- `attention_type = "mha"`
- `attention_type = "gqa"`

当使用 GQA 时，例如：

- `num_q_heads = 8`
- `num_kv_heads = 4`

表示：

- Query 头有 8 个
- Key / Value 头只有 4 个
- 每 2 个 Query 头共享 1 组 Key / Value

实现顺序严格为：

1. 线性映射得到 `q / k / v`
2. reshape 成多头张量
3. 对 self-attention 的 `q / k` 使用 RoPE
4. 再对 `k / v` 执行 GQA 的 head repeat
5. 计算 attention

### 5.4 Teacher Forcing 的输入输出构造

若目标句 token 为：

```text
["i", "love", "you"]
```

训练时构造为：

- `tgt_input = ["<bos>", "i", "love", "you"]`
- `tgt_output = ["i", "love", "you", "<eos>"]`

这样 decoder 每一步都使用真实历史 token 作为条件，这就是 teacher forcing。

## 6. 训练流程说明

`train.py` 中实现了完整训练流程：

1. 前向传播 `forward`
2. 交叉熵损失计算
3. 反向传播 `backward`
4. `optimizer.step()`
5. 验证集评估
6. 保存最佳模型

优化器使用：

- `AdamW`

并包含：

- `gradient clipping`
- 可选 `label smoothing`

## 7. Mask 机制说明

项目手写了三类 mask：

1. Encoder padding mask
2. Decoder self-attention mask
3. Decoder cross-attention mask

其中 decoder self-attention mask 同时包含：

- padding mask
- causal mask

保证 decoder 在训练时无法看到未来词。

## 8. 适合作为课程实验的扩展方向

你可以在此项目基础上继续扩展：

- 使用更大的真实双语平行语料
- 加入 BLEU 等翻译指标
- 对比 `MHA` 和 `GQA` 的效果
- 对比 `ReLU / GELU / SwiGLU`
- 加入 beam search 推理
- 增加学习率调度器

## 9. 说明

本项目重点是“手动实现 Transformer 关键结构”，因此没有使用 `nn.Transformer` 作为主体。代码尽量保持模块化、可读、可教学展示，适合作为课程实验和二次修改基础。
