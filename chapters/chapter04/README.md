# 第四章：TRoC 轨迹复用

`source/` 包含 TRoC、PROPS、SAC+ReDo、MR.Q、BRO(Fast)、MAD-TD 及本章消融；`figures/` 对应图4.1—4.8。

- `source/foundation/`：策略、完整轨迹采集与共享更新实现。
- `source/troc_props/`：TRoC/消融和 PROPS 的正式执行源码与环境。
- `source/offpolicy/`：SAC+ReDo、MR.Q、BRO(Fast) 的正式执行源码与其登记环境版本。
- `source/mad_td/`：MAD-TD 的任务适配与正式执行源码，复用 TRoC/PROPS 的相同环境。

在仓库根目录执行：

```bash
python3 -m pip install -r chapters/chapter04/requirements-torch.txt
python3 chapters/chapter04/run.py
python3 chapters/chapter04/run.py --method f1 --config /path/to/config.json
python3 chapters/chapter04/run.py --method f1 --config /path/to/config.json --execute
```

方法名为 `f1`、`props`、`redo`、`mrq`、`bro`、`mad_td`、`no_history`、`no_offpolicy_correction`、`fixed_actor_lr`。配置需明确任务、种子、新输出目录和100608步训练预算；固定策略评价另加 `--evaluation-only` 并给定模型。

不加 `--execute` 只检查配置。BRO/MAD-TD 使用独立环境安装 `requirements-jax.txt`；两套 NumPy/PyTorch 版本分别保留。数据目录通过 `RESOURCE_ALLOCATION_DATA_ROOT` 指定，模型和大型数据另行提供。

图件的可编辑源按图号放在 `figures/`。数值图使用对应目录的 `plot.py --data-dir /path/to/original/figure-package --font-dir /path/to/original/fonts --output-dir /path/to/new/output`；输入和原字体需另行提供，绘图不会自动替换字体。
