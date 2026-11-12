图2.8：资源配置实例
==================

本目录是当前完整维护目录。正文插入 figure.pdf；figure.png、figure.svg 和 figure_grayscale.png 为同版本导出。
原始 CSV、来源记录和可维护 Python 源码均在本目录及子目录内，字体读取论文工程 assets/fonts/windows。

在本图目录执行离线复绘，使用全新输出目录：

    python3 plot.py --output-dir /tmp/figure_2_8_preview

plot.py 调用本地 replot_sources/renderer.py。renderer 先按本地父图与原始数据完成复现校验，再将柴油色条名称统一为“未完成需求量”；科学数据和图形几何保持。

图件有意修改并核验后，在论文根目录更新登记并编译：

    python3 scripts/verify_local_figures.py --register
    bash scripts/build.sh

source.json 和 provenance 中的历史路径仅记录来源；复绘与正文引用均使用本目录中的实体文件。

当前文件清单
------------

见本目录 CONTENTS.json：列出实际保留的CSV/数组、全部Python源码和正文PDF。
清理后的独立复绘已通过；辅助源码或校验输入即使未直接插入正文，仍属于复绘依赖。
