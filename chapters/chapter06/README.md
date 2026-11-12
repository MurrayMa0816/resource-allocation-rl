# 第六章：灌溉用水分配推演

89个区域、81日灌溉任务的环境、SeDVal方法及PASTA/COLA对比代码。

- `experiment/source/hongmiao89/`：89区任务、状态、动作与环境接口。
- `experiment/method/`：SeDVal价值预测与学习器；`experiment/execution/`、`experiment/evaluation/`：原训练与评价程序。
- `comparison/`：PASTA/COLA灌溉适配与执行程序；两法的公共实现引用 `../chapter05/source/baselines/`。
- `figures/`：论文图6.1—6.8的绘图源码、固定输入和数值核对记录；三张地图共用 `figures/shared/data/`。
- `experiment/source/_frozen/`：原科学版本绑定的物理与学习依赖；这里只保留一份。

在仓库根目录执行：

```bash
python chapters/chapter06/run.py --help
python chapters/chapter06/run.py inspect
python chapters/chapter06/run.py check
```

以上仅用Python标准库检查目录和原依赖哈希。执行科学程序前，按本章 `requirements.txt` 配置独立环境；PyTorch版本对应CUDA 12.1。

原训练与评价入口接受 `--config`，例如 `python chapters/chapter06/run.py train --config /path/to/job.json` 默认只读取配置，添加 `--execute` 才启动原程序。配置、模型权重及完整训练/DEV/TEST记录未随源码发布；正式执行仍校验原配置、源文件和数据身份。绘图还需对应字体，图源中保留原字体与版式参数。

原有软件许可见 `LICENSE`，外部气象及农业数据来源见 `THIRD_PARTY_NOTICES.rst`。
