chapter05：第五章：双目标联合配置的分项价值学习
======================================================================

本章代码、模型和结果均为实际文件。先按论文小节查，再看原始运行包。

阅读顺序
--------

1. CODE_README.rst：环境、方法和训练/评价代码入口。
2. 论文小节对照表.csv → paper/节号_标题：小节说明、模型、图表和资料。
3. 模型用途对照表.csv 或 MODELS.json：具体模型、种子、评价身份和实际路径。
4. 图表与文件对照表.csv：已采用图表的本地输入与源码。

常用目录
--------

* launcher.py：本章统一入口；inspect/plan/check只读。
* plots：可单独阅读的图源及图包对应；figures是含完整输入的可复绘图包。
* paper：所有小节直接放在此层，无小节目录套小节目录。
* paper_tables：正文活动表格原文快照与相应数据。

代码职责
--------

环境：environment;_internal/src

方法：method;ablations;baselines

训练：training;formal_execution

评价：evaluation;formal_execution

allocation23、applications、frozen和chapters开头的路径从包根定位；其他代码路径在本章内。

模型与结果
----------

* formal：90项模型、训练和210组TEST。
* supplementary：价值监督DEV诊断。

源码包与原始运行包保留必要内部结构；不为压平路径而拆散导入或训练记录。
同一模型服务多个小节只存一个实体，CSV列出包内实际路径。
原始协议的旧绝对路径与历史类名是来源证据，现行定位使用INDEX和portable_paths。
ROOT代码副本的章节说明在CODE_README.rst；此前数据说明在RESULTS_README.rst。
