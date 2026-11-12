图4.8：TRoC与固定学习率变体的策略更新幅度
================================================

本目录保存该图的原始数据、可编辑源、绘图代码及最终图片。正文插入 figure.pdf。

本地复绘入口：plot.py。请在本图目录执行：

    python3 plot.py --output-dir /tmp/figure_4_8_preview

原始或整理后的数据（同一内容只列一条当前路径）：

    plotted_points.csv
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/paper_revisions/diagnostic_simplification_20261110_011802/assets/figure48/accepted_inputs/plotted_points.csv
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/paper_revisions/diagnostic_simplification_20261110_011802/assets/figure48/data.csv
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/paper_revisions/diagnostic_simplification_20261110_011802/assets/figure48/exceedance_per_seed.csv
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/paper_revisions/diagnostic_simplification_20261110_011802/assets/figure48/exceedance_summary.csv

原始绘图源码及当前入口（同一源码只列一条当前路径）：

    plot.py
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/paper_revisions/diagnostic_simplification_20261110_011802/assets/figure48/accepted_inputs/parent_renderer.py
    render_support/fs/hy-tmp/resource_allocation_results/chapter04/paper_revisions/diagnostic_simplification_20261110_011802/assets/figure48/plot.py

render_support/fs 内的路径名保留原绘图链命名，由 plot.py 映射到一次性工作区；全部文件均为本图目录内实体文件。

图件修改并核验后，在论文根目录更新登记再编译：

    python3 scripts/verify_local_figures.py --register
    bash scripts/build.sh

source.json 的 current_* 字段列当前内容；historical_provenance 仅记录旧来源。运行所需字体保存在论文 assets/fonts。

当前文件清单
------------

见本目录 CONTENTS.json：列出实际保留的CSV/数组、全部Python源码和正文PDF。
清理后的独立复绘已通过；辅助源码或校验输入即使未直接插入正文，仍属于复绘依赖。
