# 第二章：BLoAR

高维对象间资源配置。每期可用总量固定，包含完整方法、三种结构对照和 PASPO、RPO-SAC、TD7、SimBaV2、DIME 五种基线。

- `source/allocation23/`：环境、状态、BLoAR、PPO、训练和评价；第三章复用其中相同源码。
- `source/legacy_zero_head/`：论文已采用旧模型所需的零输出头版本；其余代码共享。
- `configs/protocol.json`：训练与评价设定。
- `figures/`：论文现用图的绘制源；数值输入和原模型另附。

在仓库根目录执行：

```sh
python3 -m pip install -r chapters/chapter02/requirements.txt
python3 chapters/chapter02/run.py config
python3 chapters/chapter02/run.py check
python3 chapters/chapter02/run.py train --domain v5 --seed 0 --episodes 960 --output ./runs/chapter02-new-run
```

通信任务还需通过 `ALLOCATION23_DATA_ROOT` 指定已准备的 EuroSAT RGB 数据目录（含 `images.npy`、`split_manifest.json`、`manifest.json`）。原始来源为 [EuroSAT](https://github.com/phelber/EuroSAT)，遵循其 MIT 许可。模型、完整结果和绘图输入未随源码上传。

默认采用随机输出头；旧模型版本使用 `run.py --legacy-zero-head ...`。目录重组后的新运行记录当前源码视图 SHA，原论文模型及其完整历史身份应使用原始实验资料包，不能当作本发布版新生成的模型。基线来源署名和许可证保留在 `source/allocation23/baselines/cores/`。
