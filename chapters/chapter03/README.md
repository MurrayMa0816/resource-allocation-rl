# 第三章：ReJoP

本章实现总量与对象分配联合决策方法 ReJoP，并提供对比与消融实验。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `environment/` | 联合配置任务环境 |
| `algorithms/` | ReJoP 与对比方法 |
| `configs/` | 实验配置与依赖 |
| `data/` | 表格输入 |
| `models/` | 模型索引 |
| `results/` | 训练、评价与分析结果 |
| `scripts/` | 运行、检查与绘图入口 |
| `figures/` | 论文图件与绘图输入 |

## 使用方法

依赖文件：`configs/requirements.txt`。

在仓库根目录查看运行入口：

```bash
python3 chapters/chapter03/scripts/run.py --help
```

模型说明见 [models/index.json](models/index.json)，结果说明见 [results/index.json](results/index.json)。

本章与第二章共享部分仿真环境和 EuroSAT 数据。
