# 第四章：TRoC 轨迹复用

`source/` 包含 TRoC、PROPS、SAC+ReDo、MR.Q、BRO(Fast)、MAD-TD 及本章消融；`figures/` 对应图4.1—4.8。

- `source/foundation/`：策略、完整轨迹采集与共享更新实现。
- `source/troc_props/`：TRoC/消融和 PROPS 的正式执行源码与环境。
- `source/offpolicy/`：SAC+ReDo、MR.Q、BRO(Fast) 的正式执行源码与其登记环境版本。
- `source/mad_td/`：MAD-TD 的任务适配与正式执行源码，复用 TRoC/PROPS 的相同环境。
- `provenance/`：原正式执行源的路径与校验登记。
- `artifacts.json`：113条模型用途登记、8图2表的实际输入路径与SHA256。

两套正式环境分别绑定不同原实验实现，内部日期与 `release` 目录是模型所校验的源版本，不能当作重复副本删除。`radio` 为图像通信，`v5` 为柴油配置。

在仓库根目录执行：

```bash
python3 -m pip install -r chapters/chapter04/requirements-torch.txt
python3 chapters/chapter04/run.py
python3 chapters/chapter04/run.py --method f1 --config /path/to/config.json
```

最小配置检查示例（另存为上述JSON文件）：

```json
{"method":"f1","domain":"v5","seed":0,"steps":100608,"output":"runs/chapter04-new-run"}
```

不加 `--execute` 只检查字段，不构造环境或载入模型。开始新训练才加 `--execute`；复现原结果应使用原登记配置。方法名为 `f1`、`props`、`redo`、`mrq`、`bro`、`mad_td`、`no_history`、`no_offpolicy_correction`、`fixed_actor_lr`。固定策略评价另加 `--evaluation-only` 并给定模型。

BRO/MAD-TD 使用独立环境安装 `requirements-jax.txt`，与Torch环境分开。完整资料根目录由 `RESOURCE_ALLOCATION_DATA_ROOT` 指定；通信任务还需完整包的 `datasets/eurosat_rgb/` 像素数据。

113条模型登记包含100条正式端点、5条补充PROPS端点及8条学习率DEV诊断端点；其中10条ReJoP正式参照复用第三章的同一实体。权重、原配置、训练/DEV/TEST记录与数值图输入未发布，当前没有附件下载地址。`artifacts.json` 的 `full_package_path` 相对于原完整包，不能当成本仓库路径。

图件的可编辑源按图号放在 `figures/`。数值图使用对应目录的 `plot.py --data-dir /path/to/current-figure-package --font-dir /path/to/assets/fonts/windows --output-dir /path/to/new/output`；加 `--check` 仅核验文件，不绘图。每图 `inputs.json`/`support.json` 列明当前图包、原数值输入、父图与版式校验附件；当前定稿图包与旧完整实验包不是同一目录，不能只补CSV。字体不会自动替换。
