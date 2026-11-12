本图独立复绘
==============

正文引用本目录的 figure.pdf。plot.py 为本地入口，具体数据与原始代码路径见 README.rst 和 source.json 的 current_* 字段。

运行：python3 plot.py --output-dir /tmp/本图复绘
默认输出 replot_output，保留已验收图片。需要 Python、NumPy、Matplotlib、Pillow、PyMuPDF，以及论文 assets/fonts 内字体。

render_support/fs 保留当前绘图链实际需要的本地源码和数据；历史路径由入口映射到一次性工作区，禁止读取原结果或相邻代码工程。全部保留文件的登记见论文根目录 docs/figure_localization_20261110/FIGURE_ASSETS.json。
