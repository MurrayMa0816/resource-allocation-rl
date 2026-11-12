图3.6现有条件评价展示调整
========================

正文保留ReJoP、去总量学习及各域外部参考三种方法。通信外部参考为C-PPO，柴油为Gradient PPO。删除Direct与NoTask图元只是展示范围调整；inputs/data.csv.gz完整保留父资产250行种子结果，旧辅助数值未删除或改名。

该图继续采用原有已使用的共同128情景、五个训练种子及种子间样本标准差，不是新增独立检验。未对随机对象分配追加条件实验。新图36聚合行见display_aggregates.csv，150条唯一种子记录见display_seed_data.csv；完整原数据入口为data.csv.gz。full_minus_pace.csv列出配对差。

绘图先逐像素复现当前打印版父PNG，再删除旧对照图元；保留的12条曲线、全部误差条、坐标范围/刻度/物理位置/单位、156×157mm画布和10.5bp字体均保持。仅同步ReJoP与去总量学习短图例，并重排共同图例。

复绘：python -B plot.py --output-dir NEW_HY_DIRECTORY。输出与缓存均必须位于/hy-tmp。全过程不构造环境、不加载模型、不推理或训练、不编译论文。
