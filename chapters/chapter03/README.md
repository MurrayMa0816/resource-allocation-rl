# 第三章：ReJoP

联合学习当期投入总量与对象间分配。保留完整 ReJoP、去总量学习、随机对象分配，以及 C-PPO、PFO、Gradient PPO、RPO、PPO+ 五种比较器。`radio` 为图像通信，`v5` 为柴油配置。

- `source/allocation23/`：本章公共状态、全额配置约束与五种比较器。
- `source/native/`：ReJoP 实际随机输出头、训练及条件评价代码。
- `source/random_object/`：随机对象分配的 DEV 消融及其必要父策略。
- `configs/protocol.json`：固定步数、种子和评价协议。
- `figures/`：图3.1—3.8的当前可编辑源与输入清单。
- `artifacts.json`：80条模型用途登记、8图3表的实际输入路径与SHA256。

本章入口明确依赖 `../chapter02/source/source_runtime.py` 和第二章中的共用环境、基础源码，均只保存一份。`source/views.json` 指定各实验所用的实际版本，保留原 Python 包名 `allocation23`。

在仓库根目录执行：

```sh
python3 -m pip install -r chapters/chapter03/requirements.txt
python3 chapters/chapter03/run.py config
python3 chapters/chapter03/run.py check
python3 chapters/chapter03/run.py train --domain v5 --arm A_FULL --seed 0 --steps 100608 --output ./runs/chapter03-new-run
```

`baseline train|evaluate` 调用五种比较器；PFO 依赖的 `spo.py` 提供采集和 GAE 基类，并非额外比较器。`random-object train` 仅对应已采用的 DEV 消融，不产生新的独立 TEST 证据。

通信任务复用第二章 `data/eurosat_rgb/` 的索引与外部像素数组，按第二章说明设置 `ALLOCATION23_DATA_ROOT`，不另复制一套数据。

原完整包的80条模型登记中，70条为正式采用，10条随机对象分配模型仅用于DEV消融；随机对象模型没有新增独立TEST。图3.4—3.5的DEV统计、主比较确认结果与图3.6—3.8的条件/实例不能混用。

模型、完整训练记录、数值图输入和表格快照尚未发布，目前没有附件下载地址。`artifacts.json` 的 `full_package_path` 相对于原完整实验包，索引不代表已经上传文件。图源还需对应原图包和原字体。

各数值图的 `plot.py` 支持 `--data-dir`（当前完整图包）、`--font-dir`（原 `assets/fonts/windows`）、`--output-dir` 与 `--check`。预检会核对 `local_inputs_sha256.json` 的全部输入；其中包含父图和版式参照，不能只提供CSV。

新运行记录本发布版源码视图。原生条件评价保留原40个端点封存协议，需原始配置、模型与资料才能执行。第三方署名和许可证随比较器源码保留。
