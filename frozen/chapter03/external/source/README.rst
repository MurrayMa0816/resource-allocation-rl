资源配置统一维护工程
====================

唯一开发入口为 ``/root/gamewar/resource_allocation_rl/``，各章在 ``chapters/`` 下分别维护。
第二章与第三章已分为两个独立工作目录，与第四、第五章并列。

章节位置
--------

* ``chapters/chapter02/``：第二章固定总量配置；本章A、三项空间组件消融、五个对比方法、配置及运行入口。
* ``chapters/chapter03/``：第三章跨期总量决策；本章A、四项总量机制消融、五个对比方法、配置及运行入口。
* ``chapters/chapter04/``、``chapter05/``、``chapter06/``：继续使用已整合的独立章节工作区。

第二章对比方法为PASPO、RPO-SAC、TD7、SAC-SimBaV2、DIME。
第三章对比方法为C-PPO、CrossQ、Gradient PPO、Reflective PPO、PPO+。
第五章旧外部比较已退役，当前外部名单为空。

共用的物理环境、精简状态、A基础网络和PPO更新在 ``allocation23/`` 保留一份。
第二章学习总量给定时的对象配置，第三章继承对象配置思想并学习跨期总量。
章节专用源码已实际归入各自目录；原导入地址和顶层注册文件是指向同一份文件的兼容链接。

常用命令
--------

在工程根运行::

    ./project check
    ./project chapter2 inspect
    ./project chapter3 inspect
    ./project chapter2 train --help
    ./project chapter3 train --help
    ./project chapter2 baseline train --help
    ./project chapter3 baseline train --help

各章的 ``launcher.py`` 也可通过本工程 ``./python`` 单独执行。
训练和评价必须显式给出新HY输出目录；inspect/help不启动训练。
第三章原 ``./python_ch3`` 和两章原 ``./python run.py`` 保持兼容。
第三章 ``./project chapter3 train`` 默认本章A，``baseline`` 子命令进入外部方法。

源码实体和新增缓存放在 ``/hy-tmp/resource_allocation_project_code_20261105/``，
通过主工程 ``chapters/`` 直接编辑，不另维护副本。
结果放 ``/hy-tmp/resource_allocation_results/chapterN/独立实验名/``，不混入源码。

完整说明见 ``docs/整合后使用说明_20261105.rst`` 及各章README。
``MAINTAINED_CODE.json`` 是当前维护登记；``CHAPTER2_CURRENT.json``、``CHAPTER3_CURRENT.json``
分别链接各章当前配置。历史执行源码与科研结果保留原位。
``chapter23_final`` 保留为历史参考，灌溉原件继续保护。

工程检查::

    ./python scripts/check_project.py
    ./python scripts/check_chapter3.py

分目录和短工程门禁不等于新增正式性能比较；正式训练仍应在独立范围冻结协议和源码。
