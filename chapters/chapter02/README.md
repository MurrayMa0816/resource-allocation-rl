# 第二章：BLoAR

本章保存高维动作表示方法、三种结构对照和五种比较方法。`radio`为图像通信，`v5`为柴油配置。

- `environment/`：任务、状态、物理环境与固定配置。
- `algorithms/`：BLoAR、PASPO、RPO-SAC、TD7、SimBaV2、DIME。零输出头模块只为论文已采用模型保留，不另放整套旧工程。
- `data/`：EuroSAT像素数组、图像顺序、划分与来源许可；第三章引用同一实体。
- `models/`：210条采用模型的唯一路径与用途清单。
- `results/`：训练、DEV、独立TEST、结构与条件分析原记录。
- `configs/`：正式协议、依赖清单与逻辑源码映射。
- `scripts/`：训练/评价/基线入口与文件查找；`figures/`为当前论文图源和实际输入。

在完整包根执行 `python3 chapters/chapter02/scripts/run.py check`。`config`查看协议，`--help`查看入口；二者不载入模型。主比较和结构对照约15万步（通信4688回合，柴油6256回合），条件分析960回合。原研究的多种子、已采用阶段和方法变体分别保留，不混作同一模型。

第二、三章相同环境只在本章保存一份。`configs/source_views.json`明确每个逻辑模块的实际位置与SHA，第三章只保其不同实现。

模型用途与唯一路径见 `models/index.json`；原结果分组见 `results/index.json`。`python3 chapters/chapter02/scripts/check.py` 核对全部登记文件，添加 `--hash` 才读取大文件逐SHA检查。图源输入由 `scripts/figure.py --number 2.1 --check` 检查，`figures/inputs/` 保存校验值唯一的当前复绘依赖。

完整权重、数组和原始记录已发布在本仓库 [Release v2026.11.19](https://github.com/MurrayMa0816/resource-allocation-rl/releases/tag/v2026.11.19)。此网页目录用于浏览源码；请用根目录 `download_materials.py` 下载并还原完整包后，再运行文件完整性检查。GitHub自动生成的源码ZIP不包含Release附件。
