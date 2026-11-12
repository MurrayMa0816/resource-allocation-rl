第二章：高维资源配置的低维动作表示
====================================================================

本目录为当前维护入口，不保存模型和训练结果副本。代码职责与论文小节.csv按环境、方法、基线、训练、评价、消融列出项目内实际代码。

共享Python包保持allocation23名称和完整结构，避免改变已存模型类名。不可变实际执行源实体在本项目frozen/chapter02，每份科学文件字节及原身份保持；维护入口优先本项目源码，不从另一份代码工程隐式导入。

2.1环境与公开状态：allocation23/environments、state.py、task.py。2.2 BLoAR科学实现：frozen/chapter02/native_random_head/allocation23，由method.py/native_backend.py核SHA后使用。五对比方法：allocation23/baselines，PASPO/RPO-SAC/TD7/SimBaV2/DIME；accepted模型仍按真实legacy_zero_head源身份复核，不将兼容科学源视作废弃实验。

2.3三种结构对照在ablations.py，条件和规模以最终MODELS及完整JSON为准。默认960回合阶段与十五万步端点是两个真实模型及两套TEST；配置实例复用端点TEST path0。

常用命令（在项目根运行）::

    ./python ./project chapter2 inspect
    ./python ./project chapter2 check
    ./python ./project chapter2 baseline inspect

inspect/status/plan只读现成索引，check只核源码SHA，均不载模型或创建环境。train/evaluate仅在用户显式调用时进入科学程序；本次整理未执行。历史dispatch/watchdog/controller/旧队列仅作封存来源，不自动恢复。

plots保存当前实际绘图代码和每图输入包对照；最终模型、TRAIN/DEV/TEST原始数据、完整图包统一由final_data_root与chapter_data_root定位。在HY完整实验版，这些实体同处一个项目目录；ROOT代码版只保全部代码、配置、许可和源快照，数据通过显式RESOURCE_ALLOCATION_DATA_ROOT指定。
