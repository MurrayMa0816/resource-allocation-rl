# 第六章：灌溉用水分配

89个区域、81日任务。保留SeDVal三档供水各3种子共9个端点，以及PASTA、COLA在400万立方米供水下各3个端点；共15个模型。外法采用共同DEV比较，SeDVal独立TEST、供水条件和反馈实例分别登记。

| 目录 | 内容 |
|---|---|
| `environment/` | 89区环境接口与模型绑定的物理依赖源码 |
| `algorithms/` | SeDVal学习器及PASTA/COLA任务适配；两种外法公共实现归第五章 |
| `configs/` | 训练/评价配置、协议、依赖签名与来源许可 |
| `data/` | 区域、原始天气、物理模型固定输入及五张表的数值 |
| `models/` | 15个已采用固定端点，身份与路径见模型索引 |
| `results/` | 完整训练、DEV/TEST、轨迹与最终审计，按SeDVal/PASTA/COLA分组 |
| `scripts/` | 唯一检查入口与显式执行程序 |
| `figures/` | 当前图6.1—6.8；图6.4由两幅成对图展示四个阶段 |

```bash
python3 chapters/chapter06/scripts/run.py inspect
python3 chapters/chapter06/scripts/run.py check
```

这两条命令只读小源码与元数据，不加载模型或运行环境。第三方依赖见 `configs/requirements.txt`；Torch环境独立配置。

科学源码按职责单份存储，`configs/source_views.json` 保存模型绑定的原逻辑路径与SHA256。原93区物理文件是当前89区模型使用的底层输入，不是另一个待运行版本。`configs/source_identity/protocol/selected_application.json` 是原签名证据，其中历史状态不代表当前仍待训练。

需要显式执行程序时，先用 `python3 chapters/chapter06/scripts/run.py capsule --output /hy-tmp/new-run/source` 在新目录重建原字节源码胶囊，再为新运行登记配置与输出；`train/evaluate/comparison --config ...` 默认只读取配置，只有加 `--execute --source-capsule ...` 才派发原程序。已完成的原配置保留为来源，不自动恢复或重放。临时胶囊全部是普通文件，运行结束后可清理；维护包不常驻多套副本。

原始数值、模型权重及DEV/TEST身份保持，不因目录重组改变。已有结果未重新训练或评价。

模型用途与唯一路径见 `models/index.json`；原结果分组见 `results/index.json`。`python3 chapters/chapter06/scripts/check.py` 核对全部登记文件，添加 `--hash` 才读取大文件逐SHA检查。图源输入由 `scripts/figure.py --number 6.1 --check` 检查，`figures/inputs/` 保存校验值唯一的当前复绘依赖。

完整权重、数组和原始记录已发布在本仓库 [Release v2026.11.19](https://github.com/MurrayMa0816/resource-allocation-rl/releases/tag/v2026.11.19)。此网页目录用于浏览源码；请用根目录 `download_materials.py` 下载并还原完整包后，再运行文件完整性检查。GitHub自动生成的源码ZIP不包含Release附件。
