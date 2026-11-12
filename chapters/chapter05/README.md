# 第五章：SeDVal 分项价值学习

`source/` 是正式实验使用的唯一源码；`execution/` 包含 100608 步训练和固定策略评价入口，`figures/` 对应本章图件。

- 方法与消融：`source/_internal/src/chapter5_multiobjective/`、`source/_internal/src/chapter5_cross_server_20261103/`。
- 仿真环境：`source/_internal/src/chapter5_full_spend_20261104/`。
- 对比方法：`source/baselines/`，包括 CAPQL、Q-Pensieve、COLA、MORL/D、PASTA；第六章直接引用其中的 PASTA/COLA。
- `artifacts.json`：90条模型用途登记、9图1表的实际输入路径与SHA256。

内部日期目录和 `native_source` 是正式模型绑定的源版本；`source_manifest.json` 校验整套285个源文件，不是另列一组可选实验。`radio` 为图像通信，`v5` 为柴油配置。

在仓库根目录执行：

```bash
python3 -m pip install -r chapters/chapter05/requirements.txt
python3 chapters/chapter05/run.py inspect
```

以下是**原完整运行包**的入口，需要先准备其原注册配置和合同，不是仅克隆仓库即可执行：

```bash
python3 chapters/chapter05/run.py train --config /path/to/config.json
python3 chapters/chapter05/run.py train --config /path/to/config.json --execute
python3 chapters/chapter05/run.py evaluate --config /path/to/evaluation.json --execute
```

数据根目录通过 `RESOURCE_ALLOCATION_DATA_ROOT` 指定。训练校验需 `chapters/chapter05/formal/contract.json` 与 `configs/`；固定策略评价还需原 `model_registry.json`、`training_audit.json`、评价配置/条件、任务完成记录及模型。即使不加 `--execute`，这些绑定资料也必须存在；加上才会运行科学程序。

90个端点为Full、OnlyC、OnlyJ、ScalarValue与五个外部方法，共9组×2任务×5种子。模型权重、原合同与配置、完整训练/评价记录及数值图输入尚未发布，目前没有附件下载地址。`artifacts.json` 的 `full_package_path` 相对于原完整包；模型用途与图表的实际统计输入分开登记。

图件的可编辑源按图号放在 `figures/`。数值图使用对应目录的 `plot.py --data-dir /path/to/current-figure-package --font-dir /path/to/assets/fonts/windows --output-dir /path/to/new/output`；加 `--check` 仅核验文件。`inputs.json`/`support.json` 区分当前图包与旧完整包的同值数值来源，并列出所需父图和版式参照；这些附件尚未公开下载。字体不会自动替换。
