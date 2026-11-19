# 第二章：BLoAR

本章实现面向资源配置的高维动作表示方法 BLoAR，并提供对比实验。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `environment/` | 资源配置仿真环境 |
| `algorithms/` | BLoAR 与对比方法 |
| `configs/` | 实验配置与依赖 |
| `data/` | EuroSAT、表格输入与字体 |
| `models/` | 模型权重与索引 |
| `results/` | 性能比较、条件分析与实例结果 |
| `scripts/` | 运行、检查与绘图入口 |
| `figures/` | 论文图件与绘图输入 |

## 使用方法

依赖文件：`configs/requirements.txt`。

在完整资料根目录查看运行入口：

```bash
python3 chapters/chapter02/scripts/run.py --help
```

模型说明见 [models/index.json](models/index.json)，结果说明见 [results/index.json](results/index.json)。
