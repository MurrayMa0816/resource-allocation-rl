# 第五章：SeDVal 分项价值学习

`source/` 是正式实验使用的唯一源码；`execution/` 包含 100608 步训练和固定策略评价入口，`figures/` 对应本章图件。

- 方法与消融：`source/_internal/src/chapter5_multiobjective/`、`source/_internal/src/chapter5_cross_server_20261103/`。
- 仿真环境：`source/_internal/src/chapter5_full_spend_20261104/`。
- 对比方法：`source/baselines/`，包括 CAPQL、Q-Pensieve、COLA、MORL/D、PASTA；第六章直接引用其中的 PASTA/COLA。

在仓库根目录执行：

```bash
python3 -m pip install -r chapters/chapter05/requirements.txt
python3 chapters/chapter05/run.py inspect
python3 chapters/chapter05/run.py train --config /path/to/config.json
python3 chapters/chapter05/run.py train --config /path/to/config.json --execute
python3 chapters/chapter05/run.py evaluate --config /path/to/evaluation.json --execute
```

不加 `--execute` 只校验配置。数据目录通过 `RESOURCE_ALLOCATION_DATA_ROOT` 指定；模型权重和大型数据另行提供。`source_manifest.json` 记录正式源码 SHA256。

图件的可编辑源按图号放在 `figures/`。数值图使用对应目录的 `plot.py --data-dir /path/to/original/figure-package --font-dir /path/to/original/fonts --output-dir /path/to/new/output`；输入和原字体需另行提供，绘图不会自动替换字体。
