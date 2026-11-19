# 资源配置强化学习实验

本仓库提供博士学位论文第二至第六章的方法实现、环境配置、论文图件及模型和实验结果索引。

## 目录结构

```text
chapters/
├── chapter02/  BLoAR：高维动作表示
├── chapter03/  ReJoP：总量与对象分配联合决策
├── chapter04/  TRoC：轨迹复用
├── chapter05/  SeDVal：分项价值学习
└── chapter06/  灌溉用水分配
```

各章采用相同的目录结构：

| 目录 | 内容 |
| --- | --- |
| `environment/` | 仿真环境与任务定义 |
| `algorithms/` | 本章方法与对比方法 |
| `configs/` | 实验配置与依赖说明 |
| `data/` | 数据与固定输入 |
| `models/` | 模型索引 |
| `results/` | 训练记录与评价结果 |
| `scripts/` | 运行、检查与绘图入口 |
| `figures/` | 论文图件与绘图输入 |

## 使用方法

按对应章节的 README 安装依赖并选择运行入口。在仓库根目录查看帮助，例如：

```bash
python3 chapters/chapter02/scripts/run.py --help
```

请保留完整的 `chapters/` 目录，部分环境、数据和工具通过相对路径跨章引用。

本匿名版本不包含完整模型权重和大型数据附件。运行相应实验需另行准备所需文件。
