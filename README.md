# 资源配置强化学习实验代码

本仓库提供博士学位论文第二至第六章的方法、环境、比较方法和绘图源码，按章节组织。

```text
chapters/
  chapter02/  BLoAR：高维动作表示
  chapter03/  ReJoP：跨期联合配置
  chapter04/  TRoC：轨迹复用
  chapter05/  SeDVal：分项价值学习
  chapter06/  灌溉用水分配推演
```

各章的 `README.md` 说明目录、依赖和运行入口；`artifacts.json` 按方法、种子和论文图表索引模型及实际输入。先查看对应章的命令：

```bash
python chapters/chapter02/run.py --help
python chapters/chapter03/run.py --help
python chapters/chapter04/run.py --help
python chapters/chapter05/run.py --help
python chapters/chapter06/run.py --help
```

第二、三章共用的环境和基础代码只保存在第二章；第六章比较方法引用第五章的实现。
`chapters/` 中的路径适配工具供各章共用。各章使用独立依赖环境，Torch 与 JAX 分开配置。

模型权重、大型图像数据和完整实验记录尚未发布，目前没有公开附件下载地址。
`artifacts.json` 中的 `full_package_path` 相对于原完整实验资料包，不能当成本仓库路径；`included` 与 `external_unpublished` 区分已包含和未发布文件。
第二、三章图表输入及第四、五章完整原执行包需另行准备；第六章已包含现用图表的固定数值输入。
仅查看命令帮助不启动训练、评价或模型载入。

第三方代码与数据遵循随附许可证及来源说明；本次整理不新增或改变原有授权。
