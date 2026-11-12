# 第三章：ReJoP

联合学习当期投入总量与对象间分配。保留完整 ReJoP、去总量学习、随机对象分配，以及 C-PPO、PFO、Gradient PPO、RPO、PPO+ 五种比较器。

- `source/allocation23/`：本章公共状态、全额配置约束与五种比较器。
- `source/native/`：ReJoP 实际随机输出头、训练及条件评价代码。
- `source/random_object/`：随机对象分配的 DEV 消融及其必要父策略。
- `configs/protocol.json`：固定步数、种子和评价协议。
- `figures/`：论文现用图源；模型和数值输入另附。

本章入口明确依赖 `../chapter02/source/source_runtime.py` 和第二章中的共用环境、基础源码，均只保存一份。`source/views.json` 指定各实验所用的实际版本，保留原 Python 包名 `allocation23`。

在仓库根目录执行：

```sh
python3 -m pip install -r chapters/chapter03/requirements.txt
python3 chapters/chapter03/run.py config
python3 chapters/chapter03/run.py check
python3 chapters/chapter03/run.py train --domain v5 --arm A_FULL --seed 0 --steps 100608 --output ./runs/chapter03-new-run
```

`baseline train|evaluate` 调用五种比较器；PFO 依赖的 `spo.py` 提供采集和 GAE 基类，并非额外比较器。`random-object train` 仅对应已采用的 DEV 消融，不产生新的独立 TEST 证据。

通信任务使用与第二章相同的外部 EuroSAT 数据。模型、完整训练结果和图表输入尚未上传；新运行的源码身份记录本发布版视图。原生条件评价脚本保留原有 40 个端点封存协议和校验，需要原始完整资料，未把它简化为新的评价协议。第三方署名和许可证随比较器源码保留。
