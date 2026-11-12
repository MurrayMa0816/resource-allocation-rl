图6.1原生110mm地图与同宽图例
================================

新增 native_map.pdf/svg/png 和 native_legend.pdf/svg/png。地图画布
110 × 92.523mm，内部board为102.949 × 77.010mm，与已接受图6.4逐图一致。
89个编号保持9.5bp，主要文字10.5bp；图6.1的q_t数学下标保留原生小字号。
地图仅改变纸面物理尺寸、沿用6.4的已验共同标注位置与原区域锚点。

维护
----

运行 python plot_native_map.py --output-dir OUTPUT，默认仅使用本图目录的
data/、source/display_helpers.py、native_map_20261110/ 与工程内字体。
显式 --source-dir 仅用于HY隔离预览。native_map_20261110/origin.json记录
来自6.4的公共原生绘图源和标注SHA；不存在对其他figure目录的运行依赖。
旧plot.py、PDF、CSV和source/保持；默认入口的新增调用由接入者在锁内追加。

图6.1的原颜色含义、区域边界与科学数值保持；图6.1保留q_t和水流标记，
图6.6保留90%阈值及12/65/76/92四低区轮廓。为保图6.1原水流标记选择，
先在虚拟156mm画布计算原文字符号排除区，再回到110mm原生输出；
ID、主文字及描线的point字号/线宽不按PDF缩放。

native_map_proof.json绑定89ID字框、字体、科学CSV列、实际画布/board、
图例与六输出SHA。所有复绘只读取冻结数据，不执行环境或学习过程。
