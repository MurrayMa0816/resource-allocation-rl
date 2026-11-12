开始阅读与核查
==============

本说明区分源码发布包的导航核查、补齐资料后的完整核查，以及需要科学依赖的
实验执行。所有命令均在相应导出仓库根执行。使用 Python 3.11 或更新版本。

一、只阅读源码与发布目录
------------------------

不需要安装 Torch、JAX、NumPy 或模型文件：

.. code-block:: console

   python3 release_tools.py inspect
   python3 release_tools.py lookup
   python3 release_tools.py lookup --help
   python3 release_tools.py check

``inspect`` 查看发布概况；``lookup`` 查看论文导航；``check`` 检查本次发布
导出中应有的文件与清单。它们只使用 Python 标准库，不启动环境、训练、推理或
新 TEST。请按输出区分“源码导出齐全”和“完整模型、数据已齐全”。

按论文阅读时，先打开 ``代码与论文小节对照.csv``，再进入对应章的
``README.rst``、``CODE_README.rst`` 和源码。源码阅读不要求先安装全部实验依赖。

二、补齐模型、数据与原始记录
----------------------------

完整原资料约 9.7 GB，不直接全部放入 Git。对外附件排除内部收据和未明确具备
分发依据的字体，并处理已知基础设施文本。数据附件尚未发布，当前没有可用的公开
下载地址。取得正式公布的附件后，应先核对其版本、文件清单与 SHA256。

附件应解压至同版本源码根目录，与 ``chapters/``、``datasets/``、``registry/``
等现有目录合并；解压结果不应成为 ``data/项目名/chapters/`` 一类额外套层。
正式与匿名入口使用同一源码和资料清单。不要以不同版本附件覆盖已有文件。

补齐资料后先执行发布版完整核查，再使用导航或兼容入口：

.. code-block:: console

   python3 release_tools.py check --require-data
   python3 project inspect
   python3 project check
   python3 project check --hash
   python3 lookup.py --figure 4.8
   python3 lookup.py --section 5.3.2.2
   python3 lookup.py --chapter 6 --method SeDVal
   python3 verify.py

``project inspect`` 阅读保存的章索引；``lookup.py`` 查询论文图表、小节、方法
及模型的对应关系，不载入模型。导出版本的 ``project check`` 和 ``verify.py``
是发布检查工具的兼容入口；``verify.py`` 要求配套资料齐全。
``release_tools.py check --require-data`` 明确要求并核对完整发布资料，读取的
数据量较大。

导出版本以 ``release_manifest.json`` 中 ``source_files`` 与 ``artifact_files``
记录的实际文件和 SHA256 为准。原包中的历史清单继续用于理解来源，但不代替
经过筛选、去标识处理后的发布清单；原完整包自己的 ``verify.py`` 保持原用途。
两个入口对应同一份发布清单。

缺失文件或哈希不符时，应根据结果补齐正确附件或核对版本，不能通过删除清单
条目把不完整资料当作核查通过。以上检查验证保存资料及其对应关系，不重新执行
论文实验。

三、需要执行实验时再配置科学环境
--------------------------------

软件依赖与科学执行入口见各章 ``CODE_README.rst``、配置及历史依赖记录。
源码仓库不包含系统 Python、虚拟环境二进制或 GPU 驱动。

* ``requirements.txt``：基础维护依赖，第二、三章等相关入口还需结合章说明。
* ``requirements-chapter04-torch.txt``：第四章 Torch 历史环境记录。
* ``requirements-chapter04-jax.txt``：第四章 JAX 历史环境记录。
* ``requirements-chapter05-torch.txt``：第五章 Torch 维护依赖。
* 第六章还应结合 ``applications/irrigation/`` 及章说明核对依赖和固定输入。

Torch 与 JAX 环境分别建立，不合并第四章两份清单的 NumPy、CUDA 等约束。
较长的历史包清单包含运行时曾安装的系统相关包和本地项目包，是环境来源记录，
不能直接视为适合任意机器的一条安装命令。按实际操作系统、硬件及章说明准备
环境后，再核查所选方法的配置。

训练、使用已有模型评价和依据已有观测绘图是不同操作。具体命令以对应章实际
入口的说明和帮助为准；不要把历史队列、controller、dispatch 或恢复脚本当作
快速开始命令。新实验应使用独立输出位置，保持论文原始模型和结果。

本次发布准备未重训、未重新评价模型，也未验证跨平台逐位复现；文件完整性通过
不等于已完成科学复现。读取和核查模式之外的运行，须由使用者按章说明显式启动。

四、恢复同版本的分片资料附件
----------------------------

将 DATA_PARTS.json 与全部 .part001、.part002 等文件放在同一下载目录。
无需先合并成另一份大文件，使用流式恢复命令：

.. code-block:: console

   python3 release_tools.py extract --parts-manifest data_downloads/DATA_PARTS.json
   python3 release_tools.py check --require-data

下载目录示例为 data_downloads；请按实际目录替换。
每片均校验 SHA256，附件解包只接受发布清单允许的普通文件。
