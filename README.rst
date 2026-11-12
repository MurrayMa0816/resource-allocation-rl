资源配置研究：论文第二至第六章代码与实验资料
============================================================

本项目提供论文第二至第六章涉及的环境、方法、训练和评价源码，以及按论文
小节查找模型、实验记录和图表输入的导航。源码仓库与模型、数据附件分别发布；
补齐同版本附件后，源码根目录同时作为实验资料根目录。

源码统一维护在本仓库，当前设为私有。模型与数据附件、匿名评审入口尚未发布，
本文档不提供尚不存在的数据下载链接或 DOI。仓库中的导航清单不表示大资料已随源码上传。

先从哪里开始
------------

* ``QUICKSTART.rst``：先浏览和核查，再按需要补齐数据与科学计算环境。
* ``LICENSE_SCOPE.rst``：第一方发布许可状态、第三方许可与数据使用范围。
* ``DATA_SOURCES.rst``：EuroSAT、第六章输入、文献与字体的来源和处理范围。
* ``PAPER_MAP.rst``：当前论文章节题名、页码与保存实验目录的对应关系。
* ``CURRENT_PAPER_UPDATES.rst``：新增表6.4、6.5的既存记录与模型入口。
* ``ANONYMOUS_REVIEW.rst``：匿名评审版本的用途、处理范围和限制。
* ``代码与论文小节对照.csv``：按论文小节找到环境、方法、训练和评价源码。
* ``chapters/chapter02`` 至 ``chapters/chapter06``：每章说明、入口与实现。
* 各章 ``paper/`` 及 ``论文小节对照表.csv``：从章节和小节进入资料。

发布目录与维护方式
------------------

统一源码位于准备目录的 ``repository/``，压缩包为 ``source.zip``。
正式仓库和匿名评审入口提供同一个固定源码版本，算法、环境、配置与资料清单相同。
匿名处理针对访问页面、账号、提交历史和身份信息，不产生另一套算法或实验版本。
本文件位于源码根时，下文所有路径均相对该根目录。

原完整实验包继续独立保留，原文件字节保持。其总量约 9.7 GB，包含模型、固定输入、
训练和评价记录等资料，不直接全部提交到 Git。对外资料附件选取其中未放入 Git
的科学资料，排除内部收据和未明确具备分发依据的字体，并对已知远程账号、地址等
基础设施文本作去标识处理；处理项随发布记录保存。源码导出与资料附件应按同一
发布版本配套。
附件下载入口尚未发布；实际发布后应同时提供附件清单、体积和 SHA256。

资料附件解压到相应源码根目录，使 ``chapters/``、``datasets/``、``registry/``
等目录与源码合并，形成统一的代码和资料目录。不要额外嵌套一个附件目录。
两个入口使用同一版本的源码及资料清单；解压前核对版本和SHA256。

第二至第六章
------------

.. list-table:: 章节导航
   :header-rows: 1
   :widths: 12 43 45

   * - 目录
     - 研究内容
     - 主要阅读入口
   * - ``chapters/chapter02``
     - BLoAR：高维资源配置的低维动作表示
     - 章说明、``environment.py``、``method.py``、``launcher.py``
   * - ``chapters/chapter03``
     - ReJoP：跨期投入与对象间分配的联合决策
     - 章说明、``method.py``、``formal_native.py``、消融实现
   * - ``chapters/chapter04``
     - TRoC：联合策略的轨迹复用与样本效率
     - 章说明、``run.py``、正式执行与方法实现
   * - ``chapters/chapter05``
     - SeDVal：双目标联合配置的分项价值学习
     - 章说明、``method/``、``training/``、``evaluation/``
   * - ``chapters/chapter06``
     - 灌溉用水分配任务的推演与方案分析
     - 章说明、``environment/``、``method/``、``evaluation/``

各章先读 ``README.rst`` 和 ``CODE_README.rst``；其中数据、模型和结果相关的
入口需要配套资料附件。``allocation23/`` 保存第二、三章共用实现，
``applications/irrigation/`` 保存灌溉应用实现，``frozen/`` 和各章冻结源码目录
保存结果对应的执行依据。保留这些内部结构有助于维持导入、模型类名和来源对应。

最小浏览与核查
--------------

使用 Python 3.11 或更新版本，在导出仓库根执行：

.. code-block:: console

   python3 release_tools.py inspect
   python3 release_tools.py lookup
   python3 release_tools.py check

这组命令用于发布内容和导航核查，仅使用 Python 标准库，不安装 Torch、JAX 等
科学计算依赖，也不读取模型进行推理。源码包通过核查不代表模型和数据附件齐全；
实际覆盖范围以命令输出为准。导出版本以 ``release_manifest.json`` 的
``source_files`` 和 ``artifact_files`` 为真实文件清单；原包历史清单不能替代
经处理后的发布清单。完整资料核查见 ``QUICKSTART.rst``。

实验身份与可复现范围
--------------------

TRAIN、DEV、TEST 分别表示记录中注明的训练、开发或诊断、独立评价身份。
同一模型的附带记录不等于某张图表的全部实际输入；应结合图表输入索引核对。
第六章固定策略换条件与改变条件后重新训练分别登记，不相互替代。

本次工作整理发布结构、说明和核查入口，没有重新训练、重新执行科学终测，
也不声称在任意操作系统或不同依赖版本下可逐位得到相同结果。具体执行入口、
配置和历史环境以相应章说明及配套记录为准。

公开分发的技术边界
------------------

GitHub 普通 Git 仓库拒绝大于 100 MiB 的文件，并建议控制仓库总体积。大模型和
原始记录采用单独附件；正式 Release 的单个附件当前要求小于 2 GiB。
具体平台额度以发布时官方文档为准。

* `GitHub 大文件说明 <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>`_
* `GitHub Release 说明 <https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases>`_
