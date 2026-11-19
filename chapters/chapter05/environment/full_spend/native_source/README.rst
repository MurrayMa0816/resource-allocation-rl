第五章实验环境依赖
=========

本目录保存第五章实验调用的资源配置环境和配套依赖。

主要内容
--------

* ``allocation23/environments/``：通信与应急资源配置任务的环境实现。
* ``allocation23/state.py``：环境观测的状态编码。
* ``allocation23/methods/``：约束映射和方法运行所需的辅助实现。

使用方法
--------

本目录由章级运行入口调用，依赖和操作说明见本章 ``README.md``。
在资料根目录查看入口参数::

    python3 chapters/chapter05/scripts/run.py --help
