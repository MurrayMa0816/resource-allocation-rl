# 第四章：TRoC

本章实现轨迹复用方法 TRoC，并提供对比与消融实验。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `environment/` | 任务环境与依赖 |
| `algorithms/` | TRoC 与对比方法 |
| `configs/` | 实验配置 |
| `data/` | 表格输入 |
| `models/` | 模型索引 |
| `results/` | 对比、消融与诊断结果 |
| `scripts/` | 运行、检查与绘图入口 |
| `figures/` | 论文图件与绘图输入 |

## 使用方法

依赖文件：`environment/requirements-torch.txt` 或 `environment/requirements-jax.txt`。Torch 与 JAX 方法分别配置运行环境。

在仓库根目录查看运行入口：

```bash
python3 chapters/chapter04/scripts/run.py --help
```

模型说明见 [models/index.json](models/index.json)，结果说明见 [results/index.json](results/index.json)。
