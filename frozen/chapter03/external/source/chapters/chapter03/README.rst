第三章独立开发目录
==================

本章的方法A训练、四项总量消融、五种外部对比及配置在本目录维护。
对比方法为C-PPO、CrossQ、Gradient PPO、Reflective PPO、PPO+。

* ``src/chapter3_candidates/``：本章唯一实体学习实现、A采集器、五法工厂及评估器。
* ``src/chapter3_candidates/profiles/``：五个外部方法的双场景配置。
* ``method.py``：本章A和对比方法的调用接口。
* ``environment.py``：本章环境接口，复用公共物理过程。
* ``ablations.py``、``configs/ablations.json``：完整A及四项总量消融。
* ``configs/current.json``、``configs/baselines.json``：当前方法和五法来源登记。
* ``launcher.py``：独立运行入口。

从总工程运行::

    ./project chapter3 inspect
    ./project chapter3 train --domain radio --arm A_FULL --steps 24576 --output /hy-tmp/resource_allocation_results/chapter3/NEW_A
    ./project chapter3 baseline train --method crossq --domain radio --steps 24576 --output /hy-tmp/resource_allocation_results/chapter3/NEW_crossq
    ./project chapter3 ablations

也可用总工程的 ``./python chapters/chapter03/launcher.py`` 运行相同命令。
旧 ``allocation23/chapter3_candidates`` 是指向本章实体源码的兼容软链接，保持Python包名与原模型类名。
后续直接修改本目录，不维护旧路径的第二套源码。

A的区域配置网络、PPO更新、物理环境和状态编码依赖公共 ``allocation23`` 基础层，保留单一实现。
本章每回合必须分配完整资源，保持原终端功率上限等物理条件。默认保存完整曲线和一个固定终点评价模型；
开发评价与独立终测分开登记。当前组织和工程验证不等于五方法已完成同预算正式比较。
