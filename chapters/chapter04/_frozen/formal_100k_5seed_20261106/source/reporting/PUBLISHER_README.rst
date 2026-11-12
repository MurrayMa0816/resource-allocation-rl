第四章验收后自动发布器
======================

调用
----

python3 execution/publish_thesis.py --scope 本scope --publish

默认或 --prepare 只在190项独立验收与聚合CSV绑定通过后生成HY暂存稿。没有科学结果时拒绝，不用工程测试值补位。--self-test 为零环境交互的工程检查，不触发论文编译。

统计与资产
----------

主结果为TRoC和五外部方法，消融为三种明确变体。A仅作共同独立终点参照。主指标使用95%固定质量要求的二连续原始DEV确认步数，未达到者保留右删失，均值与样本SD以五次训练为单位。六组图均保留本地CSV、plot.py、figure_config.json和source.json，主曲线不平滑。内部指标按每种子3072步不重叠区间汇总，不能解释为梯度方差证明。PROPS半学习率如实披露，也保留旧默认配置更好的敏感性事实。

发布范围
--------

只更新当前第四章4.3与小结、六组实验图及对应图索引行，另对章首与4.1/4.2作六处精确的全额合同同步。方法的其余数学及三份drawio不改。集成前保留整个旧第四章及旧索引和PDF，利用CAS阻止覆盖并行C4修改。其余章节改动保留。只调用scripts/build.sh，五轮均强制真实XeLaTeX整稿构建并分别检查结构、数据绑定、实际字体/渲染和数值。

最终不可变PDF保存在publication/final/main.pdf，FINAL_THESIS_PUBLICATION.json仅在全门禁和自动检查通过后写published=true。manual_visual_review保持pending，自动渲染不称为人工目视。所有准备与绘图缓存实体HY，无模型/环境/训练调用。
