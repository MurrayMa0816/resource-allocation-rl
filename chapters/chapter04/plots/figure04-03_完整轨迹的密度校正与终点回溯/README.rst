图4.3：完整轨迹的密度校正与终点回溯
======================================

本目录保存该图的原始数据、可编辑源、绘图代码及最终图片。正文插入 figure.pdf。

可编辑框图源：source.drawio。请在论文根目录导出：

    python3 scripts/export_method_figures.py --figure 4.3

图件修改并核验后，在论文根目录更新登记再编译：

    python3 scripts/verify_local_figures.py --register
    bash scripts/build.sh

source.json 的 current_* 字段列当前内容；historical_provenance 仅记录旧来源。运行所需字体保存在论文 assets/fonts。
