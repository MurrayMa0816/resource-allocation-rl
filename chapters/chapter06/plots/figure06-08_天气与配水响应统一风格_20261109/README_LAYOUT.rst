图6.8原生版面尺寸维护
================================

本次新增 layout.pdf / layout.svg / layout.png，供第六章连续排版引用。
144 × 113.760 mm；主绘图区高 42.8 mm。 字体保持原生 10.5 bp，未通过缩放既有 PDF 改变字号。

维护入口
--------

默认入口 ``python plot.py --output-dir OUTPUT`` 同时复绘原 figure.* 与新增
layout.*。原入口代码作为完整前缀保留，只在末尾增加调用。也可仅运行
``python plot_layout.py --output-dir OUTPUT``。入口只读取本目录的既有 CSV、
数值 artist 核查记录及 source/ 下既有原生绘图源；字体沿用工程内的
assets/fonts/figure/ThesisFigureText-Regular.otf。

数据与呈现
----------

原 figure.pdf / svg / png、CSV、原 source/ 与原数值记录均保留。新增入口
在创建 axes 时指定物理尺寸，再复用原 spacing.py 的字框间距规则。
逐值比较 limits、ticks、曲线、散点、误差棒、样式、标注目标数据坐标；
7/30“未调整”的显示偏移由 (0,-17) 改为 (0,-9) point，目标数据坐标保持，以避开下坐标轴。
显示偏移例外先逐项核对，再仅将该显示位置规范化回父记录进行全量比较。

layout_geometry.json 记录实际 PDF 尺寸、原生绘图区尺寸、字框间距、
原图与 CSV SHA256、输出 SHA256 以及显示例外；layout_numeric_artists.json
保留本次绘图前后相同的 artist 数值记录。输出 PDF 取消生成时间元数据，
新三种格式可以逐字复绘。仅执行既有数据绘图，不加载环境、策略或模型。
