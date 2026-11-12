图2.2：BLoAR的总体框架
================================================================

本目录是该图的当前完整维护目录，图源、数据、绘图代码及正文 PDF 均为本地实体文件。
正文插入：figure.pdf。

可编辑源为 source.drawio。请在论文工程根目录执行：

    python3 scripts/export_method_figures.py --figure 2.2

图件修改完成并核验后，在论文根目录显式更新登记，再编译：

    python3 scripts/verify_local_figures.py --register
    bash scripts/build.sh

source.json 及 provenance 中的历史绝对路径仅作来源记录；当前绘图与编译使用本论文工程中的实体文件。
