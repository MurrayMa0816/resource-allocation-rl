资源配置统一工程
================

本目录当前版本为1.1.0。第二章和第三章采用已确认的精简状态、新应急环境和图像下行传输环境。
后续研究主方法只使用A，并与五种外部算法比较。A的PPO训练与物理过程沿用已核验实现。B仅保留历史检查点兼容能力，不进入新的实验矩阵。
旧通信、库存和旧F方法已经退出活动入口。第四、第五章尚未在此整合，不以旧候选充当新实现。

当前目录
--------

* ``allocation23/environments/`` 为两个方法章环境的唯一活动实现。
* ``allocation23/methods/`` 提供A及其变体、第三章总量分支和历史B兼容读取、约束映射和独立价值网络。
* ``allocation23/state.py`` 为当前精简输入编码。
* ``allocation23/baselines/`` 保留第二章外部基线与共享物理适配，算法训练按各自原有更新规则执行。
* ``allocation23/chapter3_baselines/`` 为第三章五种在线、从零学习外部基线的活动实现，身份登记在 ``CHAPTER3_BASELINES.json``。
* ``configs/quick.json``、``formal.json`` 和 ``ablations.json`` 为仅A的已实现运行子集，配置存在不代表实验已经完成。
* ``applications/irrigation/`` 原样保留灌溉应用、物理数据、来源与许可证。
* ``environment`` 为灌溉应用的兼容别名，继续满足旧灌溉引用路径，不包含旧通信或库存环境。
* ``docs/design/`` 保留既有设计交付资料，``docs/history_20261023/`` 保留历史研究说明。

入口
----

先检查当前任务::

    ./python run.py inspect
    ./python run.py chapter2 inspect
    ./python run.py chapter3 inspect

在新的外部输出目录训练A，例如::

    ./python run.py train --domain v5 --chapter 2 --method A --episodes 960 --output /hy-tmp/my_new_v5_ch2_A
    ./python run.py train --domain radio --chapter 3 --method A --episodes 3840 --output /hy-tmp/my_new_radio_ch3_A

第二章基线入口为::

    ./python run.py baseline --help
    ./python run.py baseline inspect --domain v5 --chapter 2

第三章独立基线入口为::

    ./python_ch3 inspect --domain radio
    ./python_ch3 train --help

第三章活动方法为CAL、CCPO(NeurIPS)、C-PPO(ECAI)、Safe SAC和SafeDreamer(OSRP-Lag)。
完整来源、适配边界、五轮速度检查和运行参数见
``docs/chapter3_baseline_reselection_20261104/FINAL_REPORT.rst``。
SafeDreamer入口自动使用登记的独立JAX环境；其他四法使用现有Torch环境。

工程检查为::

    ./python scripts/check_project.py
    ./python -m pytest -q tests

``./python`` 默认使用.runtime-python登记的现有解释器，可用RESOURCE_ALLOCATION_PYTHON覆盖。
解释器必须具备requirements.txt所列依赖，灌溉支持无Numba的NumPy回退。
所有训练、评估和测试产物放外部新目录，不写回历史实验或源代码目录。

章节与比较范围
--------------

第二章每期固定额度，第三章允许自主决定总量并留存资源。两章共用空间方法和物理环境。
外部基线第二章使用固定总量；第三章新五法接收相同公开观测和总量/对象连续控制，
通过已有公共可行解码器执行，不强制使用A的预算参照头。
各算法保留自身更新机制，原论文约束与本章硬资源预算的区别明确记录。
迁移短测只验证可运行与约束，不代表960回合正式外部性能比较已经完成。

方法与边界见 ``docs/方法说明.rst``、``docs/环境与状态说明.rst`` 和 ``docs/迁移记录_20261101.rst``。
旧总工程完整归档和逐文件校验位于本轮迁移记录所列路径。
chapter23_final仍保留为独立1.0.0参考包，不在本轮修改，历史实验结果和既有论文数据作为原始记录保留。

正式训练准备度
--------------

第二章仍采用PASPO、RPO-SAC、TD7、SAC--SimBaV2及DIME，其现有正式队列不由本轮修改。
第三章新五法以 ``CHAPTER3_BASELINES.json`` 与 ``CHAPTER3_RELEASE.json`` 为准，
本轮完成实现和工程验收，尚未运行新名单的正式长训练，不能沿用旧方法曲线。

2026年11月2日的560任务方案及旧第三章对比名单，现作为历史来源保留在
``experiments/chapter23_completion_20261101/`` 与 ``configs/comparison_chapter3_a_only.json``。
活动比较CLI拒绝以旧共享名单启动第三章；历史结果读取与第二章学习核保留。

2026年11月1日的195项测试和28项配置预检是当时七方法版本的历史工程证据，
verification/formal_preparation_20261101.json不覆盖11月2日的新矩阵及容量状态。
旧checkpoint仍按实际源码身份处理，不绕过检查加载为新实验。


第三章全额分配更新（2026年11月4日）
----------------------------------

第三章当前采用每回合总资源全部分配的任务口径。A、五种外部对照与四个消融统一
使用公开可完成性下限和原物理上限，保证余额可在余下期数内合法分完，最后一期
自然固定总量。通信保留已完成终端零功率及各终端功率上限，没有放宽物理条件。
CCPO为固定预算VVE/CVI特例，C-PPO的SPF成本按解析全额时域上界归一化，
均明确登记任务适配，不声称原论文的多预算或额外安全优势。
新说明见docs/chapter3_full_spend_20261104/方法与执行说明.rst。
旧上限预算第三章模型已按授权退役，旧曲线和来源仅作历史记录，不能混入新结果。
