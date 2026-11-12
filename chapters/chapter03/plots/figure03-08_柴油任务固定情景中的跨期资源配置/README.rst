图3.8：柴油任务固定情景中的跨期资源配置
================================================================

本目录是该图的当前完整维护目录，图源、数据、绘图代码及正文 PDF 均为本地实体文件。
正文插入：figure.pdf。

本地复绘入口为 plot.py；辅助源码及数据保留在本目录子目录。
请在本图目录执行，选择独立预览输出目录以保留已验收图件：

    python3 plot.py --output-dir /tmp/figure_3_8_preview

图件修改完成并核验后，在论文根目录显式更新登记，再编译：

    python3 scripts/verify_local_figures.py --register
    bash scripts/build.sh

source.json 及 provenance 中的历史绝对路径仅作来源记录；当前绘图与编译使用本论文工程中的实体文件。

当前文件清单
------------

见本目录 CONTENTS.json：列出实际保留的CSV/数组、全部Python源码和正文PDF。
清理后的独立复绘已通过；辅助源码或校验输入即使未直接插入正文，仍属于复绘依赖。
