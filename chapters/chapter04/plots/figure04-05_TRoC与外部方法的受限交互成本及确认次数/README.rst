图4.5：各方法的训练步数统计与确认次数
========================================

本目录保存该图的原始数据、可编辑源、绘图代码及最终图片。正文插入 figure.pdf。

本地复绘入口：plot.py。请在本图目录执行：

    python3 plot.py --output-dir /tmp/figure_4_5_preview

原始或整理后的数据（同一内容只列一条当前路径）：

    render_support/fs/hy-tmp/resource_allocation_results/chapter04/figures/figure45_plain_lr_explanation_20261109_151035/assets/figure4-5/data.csv
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/figures/figure45_plain_lr_explanation_20261109_151035/assets/figure4-5/figure_config.json
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/figures/figure45_plain_lr_explanation_20261109_151035/assets/figure4-5/plotted_points.csv

原始绘图源码及当前入口（同一源码只列一条当前路径）：

    accepted_inputs/accepted_cache_adapter.py
    plot.py
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/figures/figure45_plain_lr_explanation_20261109_151035/assets/figure4-5/plot.py
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/figures/figure45_plain_lr_explanation_20261109_151035/assets/figure4-5/provenance/unused_parent_presentation_style.py
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/figures/figure45_plain_lr_explanation_20261109_151035/assets/figure4-5/replot.py
    render_support/fs/hy-tmp/resource_allocation_results/thesis_figure_spacing_unification_20261110_000617/assets/figure45/plot.py
    render_support/fs/root/gamewar/resource_allocation_rl/tools/paper_experiments/visualization/spacing_20261110/plot_chapter45.py

render_support/fs 内的路径名保留原绘图链命名，由 plot.py 映射到一次性工作区；全部文件均为本图目录内实体文件。

图件修改并核验后，在论文根目录更新登记再编译：

    python3 scripts/verify_local_figures.py --register
    bash scripts/build.sh

source.json 的 current_* 字段列当前内容；historical_provenance 仅记录旧来源。运行所需字体保存在论文 assets/fonts。

当前文件清单
------------

见本目录 CONTENTS.json：列出实际保留的CSV/数组、全部Python源码和正文PDF。
清理后的独立复绘已通过；辅助源码或校验输入即使未直接插入正文，仍属于复绘依赖。
