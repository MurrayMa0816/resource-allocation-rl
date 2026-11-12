资源配置强化学习实验代码
================================================

本仓库包含博士学位论文第二至第六章的方法实现、仿真环境、实验配置及结果索引。

目录
----

.. code-block:: text

   chapters/
     chapter02/             BLoAR：高维动作表示
     chapter03/             ReJoP：跨期联合配置
     chapter04/             TRoC：轨迹复用
     chapter05/             SeDVal：分项价值学习
     chapter06/             灌溉用水分配推演
   allocation23/            第二、三章共用环境与方法
   applications/irrigation/ 灌溉任务适配
   frozen/                  固定版本的执行源码
   registry/                论文、模型及图表索引

各章的 ``_frozen/`` 目录同样包含运行所需源码。

快速查看
--------

使用 Python 3.11 或更高版本，在仓库根目录执行：

.. code-block:: console

   python3 release_tools.py inspect
   python3 release_tools.py lookup --figure 4.8
   python3 release_tools.py check

三条命令依次查看资料概况、查找图4.8对应文件、检查本地文件完整性。
这些操作仅使用 Python 标准库；``check`` 会报告缺失的模型或数据。

运行实验
--------

安装与操作顺序见 `QUICKSTART.rst <QUICKSTART.rst>`_，具体训练、评价及绘图
入口见各章 ``CODE_README.rst``。依赖按章节安装，Torch 与 JAX 环境分别配置。

模型权重和大型数据附件尚未随仓库发布，所需数据及来源见
`DATA_SOURCES.rst <DATA_SOURCES.rst>`_。
论文对应关系见 `PAPER_MAP.rst <PAPER_MAP.rst>`_，许可范围见
`LICENSE_SCOPE.rst <LICENSE_SCOPE.rst>`_。
