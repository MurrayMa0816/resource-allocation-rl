# 第二章：BLoAR

高维对象间资源配置。每期可用总量固定，包含完整方法、三种结构对照和 PASPO、RPO-SAC、TD7、SimBaV2、DIME 五种基线。任务名 `radio` 为图像通信，`v5` 为柴油配置。

- `source/allocation23/`：环境、状态、BLoAR、PPO、训练和评价；第三章复用其中相同源码。
- `source/legacy_zero_head/`：论文已采用旧模型所需的零输出头版本；其余代码共享。
- `configs/protocol.json`：主比较、条件分析及评价设定。
- `data/eurosat_rgb/`：图像顺序、划分、校验清单与许可；不含像素数组 `images.npy`。
- `figures/`：图2.1—2.8的当前可编辑源与输入清单。
- `artifacts.json`：210条模型用途登记、8图4表的实际输入路径与SHA256；权重和原始记录未发布。

在仓库根目录执行：

```sh
python3 -m pip install -r chapters/chapter02/requirements.txt
python3 chapters/chapter02/run.py config
python3 chapters/chapter02/run.py check
python3 chapters/chapter02/run.py train --domain v5 --seed 0 --episodes 6256 --output ./runs/chapter02-new-run
```

论文主比较与结构对照使用约15万步：通信4688回合（150016步），柴油6256回合（150144步）；条件分析使用960回合，二者不能混作相同训练预算。以上训练命令会创建新运行，不恢复论文原模型。

通信任务需通过 `ALLOCATION23_DATA_ROOT` 指定含 `images.npy`、`split_manifest.json`、`manifest.json` 的数据目录。像素源为 [EuroSAT](https://github.com/phelber/EuroSAT)，MIT许可；重建数组须遵守 `data/eurosat_rgb/image_records.json` 的图像顺序与像素校验，不能任意排序替换。

模型、完整训练/DEV/TEST记录、数值图输入和表格快照尚未发布，当前没有附件下载地址。`artifacts.json` 中 `full_package_path` 相对于原完整实验包根目录，`repository_paths` 才是本仓库路径。图源需要清单对应的原图包与原字体，不能只用模型权重代替图表实际输入。

数值图入口为各图目录的 `plot.py`，参数是 `--data-dir`（当前完整图包）、`--font-dir`（原 `assets/fonts/windows`）和 `--output-dir`（新目录）。加 `--check` 仅核对清单与字体；原图包需包含 `local_inputs_sha256.json` 列出的父图和版式参照，不是只有CSV。

默认采用随机输出头；旧模型版本使用 `run.py --legacy-zero-head ...`。目录重组后的新运行记录当前源码视图 SHA，原论文模型及其完整历史身份应使用原始实验资料包，不能当作本发布版新生成的模型。基线来源署名和许可证保留在 `source/allocation23/baselines/cores/`。
