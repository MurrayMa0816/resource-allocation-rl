# 第六章：灌溉用水分配

本章将 SeDVal 应用于灌溉用水分配，提供不同供水条件下的结果及方法比较。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `environment/` | 灌溉任务环境与依赖 |
| `algorithms/` | 供水分配策略与对比方法 |
| `configs/` | 实验配置与依赖 |
| `data/` | 灌区、气象、作物及表格输入 |
| `models/` | 模型权重与索引 |
| `results/` | 供水情景与方法比较结果 |
| `scripts/` | 运行、检查与绘图入口 |
| `figures/` | 论文图件与绘图输入 |

## 使用方法

依赖文件：`configs/requirements.txt`。

在完整资料根目录查看运行入口：

```bash
python3 chapters/chapter06/scripts/run.py --help
```

模型说明见 [models/index.json](models/index.json)，结果说明见 [results/index.json](results/index.json)。

PASTA、COLA 对比方法引用第五章实现。
