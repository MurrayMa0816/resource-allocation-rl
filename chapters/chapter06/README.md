# 第六章：灌溉用水分配推演

89个区域、81日灌溉任务的环境、SeDVal方法及PASTA/COLA对比代码。

- `experiment/source/hongmiao89/`：89区任务、状态、动作与环境接口。
- `experiment/method/`：SeDVal价值预测与学习器；`experiment/execution/`、`experiment/evaluation/`：原训练与评价程序。
- `comparison/`：PASTA/COLA灌溉适配与执行程序；两法的公共实现引用 `../chapter05/source/baselines/`。
- `figures/`：图6.1—6.8的当前绘图源码与固定数值输入；图6.4含四阶段续图，共11个图块，三张地图共用 `figures/shared/data/`。
- `experiment/source/_frozen/`：原科学版本绑定的物理与学习依赖；这里只保留一份。
- `data/`：五张表的固定参数、相邻日配水、同天气供水方案和天气响应对照输入；相同物理参数仍引用 `experiment/source/`。
- `artifacts.json`：15个端点模型、8图5表的用途和输入索引。

在仓库根目录执行：

```bash
python chapters/chapter06/run.py --help
python chapters/chapter06/run.py inspect
python chapters/chapter06/run.py check
```

以上仅用Python标准库检查目录和原依赖哈希。执行科学程序前，按本章 `requirements.txt` 配置独立环境；PyTorch版本对应CUDA 12.1。

模型构成为SeDVal在350/400/450万立方米供水下分别独立训练3种子，共9个；PASTA、COLA在400万供水下各3个，共6个。外部方法只作共同DEV比较，没有三法共同的新增独立TEST；不要把全部评价和轨迹汇总当成同一评价。

本章已带地图固定输入、逐日配水与余额、四阶段状态、学习曲线、供水条件和天气响应数值。模型权重、原训练配置/合同及完整训练/DEV/TEST记录未发布，目前没有附件下载地址。`artifacts.json` 标注哪些输入已包含，其 `full_package_path` 相对于原完整实验包。

原训练与评价入口接受 `--config`，如 `python chapters/chapter06/run.py train --config /path/to/job.json`。不加 `--execute` 只读取JSON，不保证完整原运行资料齐全；添加后才运行原程序。科学执行需原合同、配置、源绑定和数据身份；原输出根限定在 `/hy-tmp`，不能用任意新JSON代替已登记配置。绘图需用户提供原字体，保留原版式参数。

各数值/地图图目录统一由 `plot.py` 绘制；图6.2使用 `create_drawio.py`。`--font-dir` 或 `THESIS_FIGURE_FONT_DIR` 指向原 `assets/fonts/` 根目录（含 `windows/`、`figure/`），`--output-dir` 指向新目录。`--help` 不导入科学库，不绘图。

原有软件许可见 `LICENSE`，外部气象及农业数据来源见 `THIRD_PARTY_NOTICES.rst`。
