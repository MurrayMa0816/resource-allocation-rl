# 第五章：SeDVal

本章实现分项价值学习方法 SeDVal，并提供对比与消融实验。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `environment/` | 资源配置任务环境与依赖 |
| `algorithms/` | SeDVal 与对比方法 |
| `configs/` | 实验配置 |
| `data/` | 表格输入 |
| `models/` | 模型权重与索引 |
| `results/` | 对比实验与价值学习分析结果 |
| `scripts/` | 运行、检查与绘图入口 |
| `figures/` | 论文图件与绘图输入 |

## 使用方法

依赖文件：`environment/requirements.txt`。

在完整资料根目录查看运行入口：

```bash
python3 chapters/chapter05/scripts/run.py --help
```

模型说明见 [models/index.json](models/index.json)，结果说明见 [results/index.json](results/index.json)。
