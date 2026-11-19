# 第5章实验资料

- `environment/`：本章实际环境与原生实现依赖；不同原实验绑定的环境分别命名。
- `algorithms/`：论文方法、采用的消融与比较方法。
- `configs/`：注册配置、当前逻辑源码与物理文件对应、SHA。
- `models/`：最终采用的种子/条件模型，实际登记见章级索引。
- `results/`：已完成训练、DEV/TEST与论文采用诊断，数据不重算。
- `scripts/run.py`：唯一章级入口；默认只检查元数据，显式执行需要新注册配置。
- `figures/`：当前论文图源和对应数值输入。

运行帮助：`python3 chapters/chapter05/scripts/run.py --help`；源检查：`python3 chapters/chapter05/scripts/capsule.py --check`。依赖清单在`environment/requirements*.txt`，Torch与JAX分环境。

`configs/source_paths.json`记录当前唯一源码视图。程序需要原相对结构时，只在临时目录用普通文件组装，并逐SHA核验后清理；原结果中的旧绝对路径只作来源身份，实际位置由章级路径索引解析。不同随机种子、消融或条件模型保留各自身份。

当前源视图清除了7个未使用的旧派发/早期门面，保留真正被100608步正式程序复用的原生worker及175环境依赖。`configs/original_model_source_manifest.json`只记录原285文件身份；当前视图278文件，不冒称原285完整副本。新实验必须登记当前source view及新合同，不能重放原完成任务。

模型用途与唯一路径见 `models/index.json`；原结果分组见 `results/index.json`。`python3 chapters/chapter05/scripts/check.py` 核对全部登记文件，添加 `--hash` 才读取大文件逐SHA检查。图源输入由 `scripts/figure.py --number 5.1 --check` 检查，`figures/inputs/` 保存校验值唯一的当前复绘依赖。

完整权重、数组和原始记录已发布在本仓库 [Release v2026.11.19](https://github.com/MurrayMa0816/resource-allocation-rl/releases/tag/v2026.11.19)。此网页目录用于浏览源码；请用根目录 `download_materials.py` 下载并还原完整包后，再运行文件完整性检查。GitHub自动生成的源码ZIP不包含Release附件。
