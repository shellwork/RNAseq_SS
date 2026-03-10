# RNAseq_SS

面向低测序深度 enrichment-based sequencing 数据的 baseline 去噪框架（参考 DeepMerip 思路）。

## 项目目标

该 baseline 对应你的 proposal，核心是：
- 用 **CNN + Transformer** 混合架构建模 coverage profile 的局部模式与长程依赖。
- 用 **多任务学习** 同时完成：
  - 回归：预测更干净的连续富集信号。
  - 分类：预测每个位点是否为 peak。
- 在 **DataLoader 中执行随机下采样**：每个 batch 随机抽取测序深度比例，让模型学习从不同低深度输入恢复原始高深度信号。

## 建议文件结构

```text
RNAseq_SS/
├── configs/
│   ├── baseline_m6a.yaml
│   └── baseline_domain_transfer.yaml
├── scripts/
│   └── train_baseline.py
├── src/
│   └── rnaseq_ss/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── losses.py
│       └── model.py
├── tests/
│   └── test_model_shapes.py
├── pyproject.toml
└── README.md
```

## Data 设计（重点：下采样）

`src/rnaseq_ss/data.py` 里新增了 `downsample_coverage`，并在 `SyntheticCoverageDataset.__getitem__` 中按样本随机选择 `depth_ratio`（如 0.1/0.2/0.5/1.0），然后对高深度信号做 Poisson thinning：

- 输入：`noisy_signal`（第 0 通道是下采样后的 noisy coverage；最后一个通道显式写入本样本 depth ratio）。
- 标签：
  - `clean_signal`：原始高深度信号（回归目标）。
  - `peak_label`：由 clean signal 导出的 peak 二分类标签。

这样可直接模拟“同一个位点在不同采样深度下观测到的信号质量变化”，对应你们项目里“随机下采样训练恢复原始信号”的核心目标。

## 快速开始

1. 安装依赖（建议在虚拟环境中）：

```bash
pip install -e .
```

2. 运行 baseline 训练：

```bash
python scripts/train_baseline.py --config configs/baseline_m6a.yaml
```

3. 跑单元测试：

```bash
pytest -q
```

## 对接真实数据时如何改

- 把 `SyntheticCoverageDataset` 替换为真实数据集类：
  - 读取原始高深度 IP/Input coverage。
  - 在 `__getitem__` 里随机采样 depth ratio，动态生成低深度输入。
  - 保留高深度信号作为监督标签。
- 你可以在 `configs/*.yaml` 中为不同 domain 设置不同 `depth_candidates`。
- 后续可增加评估指标：AUROC/AUPRC、peak-level F1、Pearson/Spearman。
