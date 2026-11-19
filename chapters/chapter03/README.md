# 第三章：ReJoP

本章保存多次分配总量与对象分配方法、去总量学习/随机对象分配消融和五种在线比较方法。

- `environment/`：本章多次分配状态、预算和约束；相同物理模块引用第二章`environment/`，不复制一份。
- `algorithms/`：ReJoP、随机对象分配、C-PPO、PFO、Gradient PPO、RPO、PPO+。
- `configs/`：100608步协议、原条件协议、源码视图及依赖清单。
- `models/`：80模型用途登记；70条正式采用、10条随机对象模型仅用于DEV消融。
- `results/`：训练、DEV、共同终点评价、条件/实例与汇总原记录。
- `scripts/`：明确的新运行入口；`figures/`保存当前论文图源及实际输入。

在完整包根执行 `python3 chapters/chapter03/scripts/run.py check`，仅检查实际源文件及SHA；`--help`查看命令。通信数据引用第二章`data/eurosat_rgb/`。每个实验使用独立进程，避免不同逻辑源码视图混用。

随机对象分配是DEV消融，没有新增独立TEST；不能用其DEV统计替换主比较的终点评价。已完成模型的原协议保留原身份，目录重组后的新运行登记当前交付源码视图。

模型用途与唯一路径见 `models/index.json`；原结果分组见 `results/index.json`。`python3 chapters/chapter03/scripts/check.py` 核对全部登记文件，添加 `--hash` 才读取大文件逐SHA检查。图源输入由 `scripts/figure.py --number 3.1 --check` 检查，`figures/inputs/` 保存校验值唯一的当前复绘依赖。

完整权重、数组和原始记录已发布在本仓库 [Release v2026.11.19](https://github.com/MurrayMa0816/resource-allocation-rl/releases/tag/v2026.11.19)。此网页目录用于浏览源码；请用根目录 `download_materials.py` 下载并还原完整包后，再运行文件完整性检查。GitHub自动生成的源码ZIP不包含Release附件。
