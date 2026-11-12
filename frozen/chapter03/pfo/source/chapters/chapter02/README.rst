第二章独立开发工作区
====================

从主工程进入 ``chapters/chapter02/``，后续第二章方法适配、消融、对比方法及配置在这里维护。
本目录实体在HY；原allocation23.baselines的学习源码已实际迁入baselines/，旧路径只保留兼容软链。

* ``method.py``：方法A和空间消融的固定总量适配。
* ``environment.py``：仅构造第二章任务的环境入口。
* ``ablations.py``：A_FULL、A_RANDOM_MATCH、A_SOFTMAX、A_DIRECT。
* ``baselines/``：五种对比方法的agent、runner与学习核心。
* ``configs/``：本章设置和既有实验计划。计划存在不代表自动启动或新正式结果。
* ``launcher.py``：本章独立运行入口。

在主工程根运行::

    ./project chapter2 inspect
    ./project chapter2 check
    ./project chapter2 train --help
    ./project chapter2 baseline train --help

也可从本目录使用主工程解释器运行 ``python launcher.py ...``。
训练和评估必须显式提供输出目录，结果与缓存实体放HY，结果不能写进维护源码树。
该入口锁定第二章、A及固定各期总量；第三章方法及跨期总量消融不从这里运行。

A基础网络、PPO更新、精简状态和两个物理环境共用主工程唯一实现，未复制成独立分叉。
现有训练器的更新和保存规则沿用，本文目录整理不把旧完整断点格式改成新的科学协议。
历史冻结任务和结果继续按原源码身份读取，新的正式训练须重新登记来源。
