第三章：跨期投入与对象间分配的联合决策
====================================================================

本目录为当前维护入口，不保存模型和训练结果副本。代码职责与论文小节.csv按环境、方法、基线、训练、评价、消融列出项目内实际代码。

共享Python包保持allocation23名称和完整结构，避免改变已存模型类名。不可变实际执行源实体在本项目frozen/chapter03，每份科学文件字节及原身份保持；维护入口优先本项目源码，不从另一份代码工程隐式导入。

3.1环境、状态与全额配置约束在allocation23/environments、state.py、full_spend.py。3.2 ReJoP完整与去总量学习分别为A_FULL/A_PACE，formal_native.py调用frozen/chapter03/native。3.3五对比方法在allocation23/chapter3_candidates，PFO需要spo.py的采集/GAE基类；保留这个依赖并不增加SPO对比。

当前消融为去总量学习与随机对象分配；random_object_ablation_20261109为后者维护入口，其10模型只有DEV证据，不声称新独立TEST。旧BudgetOnly/DirectTotal原件仅留冻结来源，不作为当前消融公开入口。70组默认确认、120组改变条件和10组补存默认轨迹保留各自身份；条件图使用150组，不重训。

常用命令（在项目根运行）::

    ./python ./project chapter3 inspect
    ./python ./project chapter3 check
    ./python ./project chapter3 baseline inspect

inspect/status/plan读取索引，check校验源码SHA，均不载入模型或创建环境。train/evaluate分别执行训练与评价。历史队列脚本保留作来源记录，不作为当前运行入口。

plots保存当前实际绘图代码和每图输入包对照；最终模型、TRAIN/DEV/TEST原始数据、完整图包统一由final_data_root与chapter_data_root定位。在HY完整实验版，这些实体同处一个项目目录；ROOT代码版只保全部代码、配置、许可和源快照，数据通过显式RESOURCE_ALLOCATION_DATA_ROOT指定。
