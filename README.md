# 资源配置强化学习：论文完整实验资料

本仓库提供第二至第六章的代码、环境、数据、模型和实验记录。完整资料共 **20,047个文件、约9.37 GB**，包含 **498个模型实体、508条用途登记、41个编号图和15张表**。

代码按 `chapters/chapter02` 至 `chapter06` 组织；每章区分 `environment`、`algorithms`、`configs`、`data`、`models`、`results`、`scripts`、`figures`，用途见各章README。

## 下载完整资料

**完整文件在本仓库的 [Release v2026.11.19](https://github.com/MurrayMa0816/resource-allocation-rl/releases/tag/v2026.11.19)，按章节分卷。** 主目录保留可浏览的源码和元数据；模型权重、数组与大型原始记录存放在Release附件。GitHub的“Download ZIP”仅下载源码，不能代替完整资料下载。

```bash
python3 download_materials.py --output /path/to/resource_allocation_final
```

下载器逐卷校验SHA256、安全解压，再核对全部文件；还原目录与本地完整包一致。需要约10 GB可用空间。若仓库仍为私有，请先 `gh auth login`，或在环境变量 `GH_TOKEN` 中提供有读取权限的凭据。凭据不会写入资料包。

已经下载的目录可离线复核：

```bash
python3 download_materials.py --output /path/to/resource_allocation_final --verify-only
```

分卷及全部文件的大小和SHA256见 `MATERIALS.json`。不同方法、种子、消融及条件模型保留各自身份；跨章使用同一模型不另存副本。

## 使用

还原完整包后，先读对应章README。帮助和检查使用Python标准库；科学程序按章安装依赖，Torch与JAX环境分开配置。新运行需要明确配置与新输出目录，不恢复已完成任务。

第二、三章相同物理环境与EuroSAT数据归第二章，第六章引用的PASTA/COLA公共实现归第五章。原模型和实验数值保持；本次发布未重新训练、评价、载模或绘图。第三方代码及数据遵循随附许可。
